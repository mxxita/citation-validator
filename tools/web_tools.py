"""
Web scraping and API tools for retrieving academic papers.

FIXES APPLIED:
1. asyncio.Lock for Semantic Scholar rate limiting (fixes 429s)
2. arXiv API as fallback (doesn't have strict rate limits)
3. Improved Semantic Scholar page scraping
4. CrossRef API for DOI lookups when SS fails
"""

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from functools import wraps
from typing import List, Dict, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from config import Config
from schemas import Reference, RetrievedSource


logger = logging.getLogger(__name__)


# =============================================================================
# CACHES
# =============================================================================
_doi_cache: Dict[str, Dict] = {}
_abstract_cache: Dict[str, str] = {}


# =============================================================================
# RATE LIMITING - FIXED WITH LOCK
# =============================================================================
_ss_lock = asyncio.Lock()
_arxiv_lock = asyncio.Lock()

async def _ss_request(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    """Semantic Scholar: serialize requests with 1s delay."""
    async with _ss_lock:
        delay = 1.0 / getattr(Config, 'SEMANTIC_SCHOLAR_RATE_LIMIT', 1.0)
        await asyncio.sleep(delay)
        return await client.get(url, params=params)

async def _arxiv_request(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    """arXiv: serialize with 3s delay (their guideline)."""
    async with _arxiv_lock:
        await asyncio.sleep(3.0)
        return await client.get(url, params=params)


# =============================================================================
# RETRY DECORATOR
# =============================================================================
def retry_async(max_attempts: int = 3, delay: float = 1.0, skip_on: tuple = (404,)):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in skip_on:
                        raise  # Don't retry 404s
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                except httpx.TimeoutException as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Timeout, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
            logger.error(f"All {max_attempts} attempts failed")
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# SEMANTIC SCHOLAR API
# =============================================================================
@retry_async(max_attempts=2)
async def search_semantic_scholar(query: str, limit: int = 10) -> List[Dict]:
    """Search Semantic Scholar API."""
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,abstract,url,externalIds,venue"
    }

    timeout = getattr(Config, 'WEB_REQUEST_TIMEOUT', 30)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await _ss_request(client, base_url, params)
        response.raise_for_status()
        data = response.json()

    results = []
    for paper in data.get("data", []):
        results.append({
            "title": paper.get("title"),
            "authors": [a.get("name") for a in paper.get("authors", [])],
            "year": paper.get("year"),
            "abstract": paper.get("abstract"),
            "url": paper.get("url"),
            "doi": paper.get("externalIds", {}).get("DOI"),
            "arxiv_id": paper.get("externalIds", {}).get("ArXiv"),
            "venue": paper.get("venue"),
            "source": "semantic_scholar"
        })

    logger.info(f"Semantic Scholar: {len(results)} results for '{query[:50]}...'")
    return results


@retry_async(max_attempts=2)
async def get_paper_by_doi_ss(doi: str) -> Optional[Dict]:
    """Retrieve paper by DOI via Semantic Scholar."""
    if doi in _doi_cache:
        return _doi_cache[doi]

    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {"fields": "title,authors,year,abstract,url,externalIds,venue"}

    try:
        timeout = getattr(Config, 'WEB_REQUEST_TIMEOUT', 30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await _ss_request(client, url, params)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            paper = response.json()

        result = {
            "title": paper.get("title"),
            "authors": [a.get("name") for a in paper.get("authors", [])],
            "year": paper.get("year"),
            "abstract": paper.get("abstract"),
            "url": paper.get("url"),
            "doi": doi,
            "venue": paper.get("venue"),
            "source": "semantic_scholar_doi"
        }
        _doi_cache[doi] = result
        return result

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise


# =============================================================================
# DBLP API (RELIABLE - no rate limits, great for CS papers)
# =============================================================================
async def search_dblp(query: str, limit: int = 5) -> List[Dict]:
    """Search DBLP - reliable for computer science papers, no rate limits."""
    base_url = "https://dblp.org/search/publ/api"
    
    # Clean query
    clean_query = re.sub(r'["\'\(\)\[\]:;]', ' ', query)
    clean_query = ' '.join(clean_query.split())
    
    params = {
        "q": clean_query,
        "format": "json",
        "h": limit,  # max hits
    }
    
    try:
        timeout = getattr(Config, 'WEB_REQUEST_TIMEOUT', 30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
        
        data = response.json()
        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        
        results = []
        for hit in hits:
            info = hit.get("info", {})
            
            # Get authors
            authors_data = info.get("authors", {}).get("author", [])
            if isinstance(authors_data, dict):
                authors_data = [authors_data]
            authors = [a.get("text", a) if isinstance(a, dict) else str(a) for a in authors_data]
            
            # Get URL (prefer DOI)
            url = info.get("ee")  # electronic edition URL
            if isinstance(url, list):
                url = url[0]
            
            doi = info.get("doi")
            
            results.append({
                "title": info.get("title", "").rstrip("."),
                "authors": authors,
                "year": int(info.get("year", 0)) if info.get("year") else None,
                "url": url,
                "doi": doi,
                "venue": info.get("venue"),
                "abstract": None,  # DBLP doesn't have abstracts
                "source": "dblp",
            })
        
        return results
        
    except Exception as e:
        logger.warning(f"DBLP search failed: {e}")
        return []


# =============================================================================
# ARXIV API (FALLBACK - more lenient rate limits)
# =============================================================================
async def search_arxiv(query: str, limit: int = 5) -> List[Dict]:
    """Search arXiv API - good fallback when SS is rate limited."""
    base_url = "https://export.arxiv.org/api/query"
    
    # Clean query - just use simple keyword search
    clean_query = re.sub(r'["\'\(\)\[\]:;,]', ' ', query)
    clean_query = ' '.join(clean_query.split())  # Normalize whitespace
    
    # Use simple "all:" search but focus on key terms
    # arXiv works best with fewer, more specific terms
    words = clean_query.split()
    # Take the most distinctive words (skip short ones and common words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'for', 'in', 'on', 'of', 'to', 'with', 'by'}
    key_words = [w for w in words if len(w) > 3 and w.lower() not in stop_words][:6]
    
    search_query = ' AND '.join(key_words) if key_words else clean_query
    
    params = {
        "search_query": f"all:{search_query}",
        "start": 0,
        "max_results": limit * 3,  # Get more results to filter
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        timeout = getattr(Config, 'WEB_REQUEST_TIMEOUT', 30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await _arxiv_request(client, base_url, params)
            response.raise_for_status()

        # Parse XML
        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        results = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns)
            summary = entry.find('atom:summary', ns)
            published = entry.find('atom:published', ns)
            
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)
            
            # Get arXiv ID from id URL
            id_elem = entry.find('atom:id', ns)
            arxiv_id = None
            url = None
            if id_elem is not None:
                url = id_elem.text
                match = re.search(r'arxiv.org/abs/(.+)', url)
                if match:
                    arxiv_id = match.group(1)
            
            year = None
            if published is not None and published.text:
                year = int(published.text[:4])
            
            results.append({
                "title": title.text.strip() if title is not None else None,
                "authors": authors,
                "year": year,
                "abstract": summary.text.strip() if summary is not None else None,
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else url,
                "arxiv_id": arxiv_id,
                "doi": f"10.48550/arXiv.{arxiv_id}" if arxiv_id else None,
                "source": "arxiv"
            })
        
        logger.info(f"arXiv: {len(results)} results for '{query[:50]}...'")
        return results

    except Exception as e:
        logger.warning(f"arXiv search failed: {e}")
        return []


async def get_arxiv_by_id(arxiv_id: str) -> Optional[Dict]:
    """Get paper by arXiv ID."""
    # Clean ID (remove version if present for search, keep for URL)
    clean_id = re.sub(r'v\d+$', '', arxiv_id)
    
    base_url = "https://export.arxiv.org/api/query"
    params = {"id_list": clean_id}

    try:
        timeout = getattr(Config, 'WEB_REQUEST_TIMEOUT', 30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await _arxiv_request(client, base_url, params)
            response.raise_for_status()

        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entry = root.find('atom:entry', ns)
        if entry is None:
            return None
        
        title = entry.find('atom:title', ns)
        summary = entry.find('atom:summary', ns)
        
        if title is None or (title.text and 'Error' in title.text):
            return None
        
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text)
        
        return {
            "title": title.text.strip() if title is not None else None,
            "authors": authors,
            "abstract": summary.text.strip() if summary is not None else None,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "arxiv_id": arxiv_id,
            "doi": f"10.48550/arXiv.{clean_id}",
            "source": "arxiv_id"
        }

    except Exception as e:
        logger.warning(f"arXiv ID lookup failed: {e}")
        return None


# =============================================================================
# CROSSREF API (for DOI lookups - no rate limit issues)
# =============================================================================
async def get_paper_by_doi_crossref(doi: str) -> Optional[Dict]:
    """Retrieve paper metadata via CrossRef (backup for DOI lookups)."""
    url = f"https://api.crossref.org/works/{doi}"
    
    try:
        timeout = getattr(Config, 'WEB_REQUEST_TIMEOUT', 30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()

        work = data.get("message", {})
        
        authors = []
        for author in work.get("author", []):
            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
            if name:
                authors.append(name)
        
        # CrossRef doesn't always have abstracts
        abstract = work.get("abstract", "")
        if abstract:
            # Clean HTML tags from abstract
            abstract = re.sub(r'<[^>]+>', '', abstract)
        
        year = None
        if "published-print" in work:
            year = work["published-print"].get("date-parts", [[None]])[0][0]
        elif "published-online" in work:
            year = work["published-online"].get("date-parts", [[None]])[0][0]
        
        return {
            "title": work.get("title", [None])[0],
            "authors": authors,
            "year": year,
            "abstract": abstract or None,
            "url": work.get("URL"),
            "doi": doi,
            "venue": work.get("container-title", [None])[0],
            "source": "crossref"
        }

    except Exception as e:
        logger.warning(f"CrossRef lookup failed for {doi}: {e}")
        return None


# =============================================================================
# UNIFIED DOI LOOKUP (tries multiple sources)
# =============================================================================
async def get_paper_by_doi(doi: str) -> Optional[Dict]:
    """Try multiple sources for DOI lookup."""
    if doi in _doi_cache:
        return _doi_cache[doi]
    
    # Try Semantic Scholar first
    try:
        result = await get_paper_by_doi_ss(doi)
        if result and result.get("abstract"):
            _doi_cache[doi] = result
            return result
    except Exception:
        pass
    
    # Try CrossRef as fallback
    try:
        result = await get_paper_by_doi_crossref(doi)
        if result:
            _doi_cache[doi] = result
            return result
    except Exception:
        pass
    
    # Try arXiv if DOI looks like arXiv
    if "arxiv" in doi.lower():
        match = re.search(r'arXiv\.(\d+\.\d+)', doi, re.IGNORECASE)
        if match:
            result = await get_arxiv_by_id(match.group(1))
            if result:
                _doi_cache[doi] = result
                return result
    
    return None


# =============================================================================
# BROWSER SCRAPING
# =============================================================================
async def scrape_abstract_simple(url: str) -> Optional[str]:
    """Extract abstract using Playwright + BeautifulSoup."""
    if url in _abstract_cache:
        return _abstract_cache[url]

    domain = urlparse(url).netloc.lower()
    browser = None
    playwright = None

    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=getattr(Config, 'BROWSER_HEADLESS', True)
        )
        page = await browser.new_page()
        
        # Set a realistic user agent
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })

        timeout = getattr(Config, 'WEB_REQUEST_TIMEOUT', 30) * 1000
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        await page.wait_for_timeout(2000)  # Wait for JS rendering

        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')

        abstract = _extract_abstract(soup, domain)

        if abstract:
            _abstract_cache[url] = abstract
            logger.info(f"Extracted abstract ({len(abstract)} chars) from {domain}")

        return abstract

    except Exception as e:
        logger.error(f"Scraping error for {url}: {e}")
        return None

    finally:
        if browser:
            await browser.close()
        if playwright:
            await playwright.stop()


def _extract_abstract(soup: BeautifulSoup, domain: str) -> Optional[str]:
    """Extract abstract with site-specific and generic selectors."""
    
    # arXiv
    if 'arxiv.org' in domain:
        elem = soup.find('blockquote', class_='abstract')
        if elem:
            text = elem.get_text()
            return text.replace('Abstract:', '').replace('Abstract', '').strip()

    # Semantic Scholar - IMPROVED
    elif 'semanticscholar.org' in domain:
        # Try multiple selectors
        selectors = [
            ('div', {'data-test-id': 'abstract-text'}),
            ('div', {'class': 'abstract__text'}),
            ('div', {'class': 'text-truncator'}),
            ('meta', {'name': 'description'}),
        ]
        for tag, attrs in selectors:
            elem = soup.find(tag, attrs)
            if elem:
                if tag == 'meta':
                    content = elem.get('content', '')
                    if len(content) > 100:  # Avoid short meta descriptions
                        return content
                else:
                    text = elem.get_text().strip()
                    if len(text) > 100:
                        return text

    # ACM
    elif 'acm.org' in domain:
        for selector in ['div.abstractSection', 'section.abstract', 'div.article__abstract']:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text().strip()

    # IEEE
    elif 'ieee.org' in domain:
        elem = soup.find('div', class_='abstract-text')
        if elem:
            return elem.get_text().strip()

    # Generic fallback
    for meta_name in ['description', 'abstract', 'DC.description', 'og:description']:
        meta = soup.find('meta', attrs={'name': meta_name}) or soup.find('meta', attrs={'property': meta_name})
        if meta and meta.get('content'):
            content = meta['content']
            if 100 < len(content) < 3000:
                return content

    for class_name in ['abstract', 'Abstract', 'abstract-content', 'summary']:
        elem = soup.find(['div', 'section', 'p'], class_=class_name)
        if elem:
            text = elem.get_text().strip()
            if 100 < len(text) < 3000:
                return text

    return None


# =============================================================================
# DOI RESOLUTION
# =============================================================================
async def resolve_doi(doi: str) -> Optional[str]:
    """Resolve DOI to final URL."""
    try:
        timeout = getattr(Config, 'WEB_REQUEST_TIMEOUT', 30)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.head(f"https://doi.org/{doi}")
            return str(response.url)
    except Exception as e:
        logger.warning(f"DOI resolution failed for {doi}: {e}")
        return None


# =============================================================================
# HIGH-LEVEL SEARCH
# =============================================================================
async def search_paper(reference: Reference) -> List[str]:
    """Search for paper and return candidate URLs."""
    candidate_urls = []
    seen = set()

    def add_url(url: Optional[str]):
        if url and url not in seen:
            candidate_urls.append(url)
            seen.add(url)

    # Direct URL
    add_url(reference.url)

    # DOI lookup
    if reference.doi:
        paper = await get_paper_by_doi(reference.doi)
        if paper:
            add_url(paper.get("url"))
        add_url(f"https://doi.org/{reference.doi}")

    # Title search - try arXiv first (no rate limit), then SS
    if reference.title:
        query = reference.title
        if reference.authors:
            query += f" {reference.authors[0]}"
        
        # Try arXiv first
        try:
            arxiv_results = await search_arxiv(query, limit=3)
            for r in arxiv_results:
                add_url(r.get("url"))
        except Exception:
            pass
        
        # Then try Semantic Scholar
        try:
            query_parts = [f'"{reference.title}"']
            if reference.authors:
                query_parts.append(reference.authors[0])
            if reference.year:
                query_parts.append(str(reference.year))
            
            ss_results = await search_semantic_scholar(" ".join(query_parts), limit=3)
            for r in ss_results:
                add_url(r.get("url"))
        except Exception as e:
            logger.warning(f"SS search failed (using arXiv results): {e}")

    return candidate_urls


# =============================================================================
# MAIN RETRIEVAL
# =============================================================================
async def get_paper_content(reference: Reference) -> RetrievedSource:
    """Retrieve paper content for a reference."""
    source = RetrievedSource(
        reference_id=reference.id,
        source_url=None,
        abstract=None,
        full_text_snippet=None,
        retrieval_success=False,
        retrieval_method=None,
        error_message=None
    )

    try:
        # Fast path: DOI lookup
        if reference.doi:
            paper = await get_paper_by_doi(reference.doi)
            if paper and paper.get("abstract"):
                source.source_url = paper.get("url")
                source.abstract = paper["abstract"]
                source.retrieval_success = True
                source.retrieval_method = paper.get("source", "doi_lookup")
                return source

        # Helper function to check title match
        def titles_match(ref_title: str, paper_title: str, threshold: float = 0.4) -> bool:
            """Check if titles have sufficient word overlap."""
            ref_title = ref_title.lower()
            paper_title = paper_title.lower()
            
            # Get significant words (4+ chars, not stop words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'for', 'in', 'on', 'of', 'to', 'with', 'by', 'from'}
            ref_words = set(w for w in re.findall(r'\w{4,}', ref_title) if w not in stop_words)
            paper_words = set(w for w in re.findall(r'\w{4,}', paper_title) if w not in stop_words)
            
            if not ref_words:
                return False
            
            overlap = len(ref_words & paper_words) / len(ref_words)
            return overlap >= threshold

        # Try DBLP first (most reliable for CS papers, no rate limits)
        if reference.title:
            query = reference.title
            if reference.authors:
                query += f" {reference.authors[0]}"
            
            dblp_results = await search_dblp(query, limit=5)
            for paper in dblp_results:
                paper_title = paper.get("title", "")
                if titles_match(reference.title, paper_title):
                    # DBLP doesn't have abstracts, but we can get the URL
                    # and scrape it, or use the DOI to get abstract
                    if paper.get("doi"):
                        doi_paper = await get_paper_by_doi(paper["doi"])
                        if doi_paper and doi_paper.get("abstract"):
                            source.source_url = doi_paper.get("url") or paper.get("url")
                            source.abstract = doi_paper["abstract"]
                            source.retrieval_success = True
                            source.retrieval_method = "dblp_doi"
                            return source
                    
                    # Try scraping the URL
                    if paper.get("url") and is_simple_scrapable_url(paper["url"]):
                        abstract = await scrape_abstract_simple(paper["url"])
                        if abstract:
                            source.source_url = paper["url"]
                            source.abstract = abstract
                            source.retrieval_success = True
                            source.retrieval_method = "dblp_scrape"
                            return source

        # Try arXiv search
        if reference.title:
            query = reference.title
            if reference.authors:
                query += f" {reference.authors[0]}"
            
            arxiv_results = await search_arxiv(query, limit=10)
            for paper in arxiv_results:
                if paper.get("abstract"):
                    paper_title = paper.get("title", "")
                    if titles_match(reference.title, paper_title):
                        source.source_url = paper.get("url")
                        source.abstract = paper["abstract"]
                        source.retrieval_success = True
                        source.retrieval_method = "arxiv_search"
                        return source

        # Get candidate URLs
        candidate_urls = await search_paper(reference)

        if not candidate_urls:
            source.error_message = "No URLs found for this reference"
            return source

        # Try scraping each URL
        for url in candidate_urls:
            if is_simple_scrapable_url(url):
                abstract = await scrape_abstract_simple(url)
                if abstract:
                    domain = urlparse(url).netloc.lower()
                    source.source_url = url
                    source.abstract = abstract
                    source.retrieval_success = True
                    source.retrieval_method = f"scrape_{domain.split('.')[0]}"
                    return source

        # Last resort: SS title search
        if reference.title:
            try:
                results = await search_semantic_scholar(reference.title, limit=3)
                for paper in results:
                    if paper.get("abstract"):
                        # Verify title similarity
                        paper_title = paper.get("title", "").lower()
                        ref_title = reference.title.lower()
                        
                        ref_words = set(re.findall(r'\w{4,}', ref_title))
                        paper_words = set(re.findall(r'\w{4,}', paper_title))
                        
                        if ref_words and paper_words:
                            overlap = len(ref_words & paper_words) / len(ref_words)
                            if overlap < 0.3:
                                logger.debug(f"Skipping SS result (low title match {overlap:.0%}): {paper_title[:50]}")
                                continue
                        
                        source.source_url = paper.get("url")
                        source.abstract = paper["abstract"]
                        source.retrieval_success = True
                        source.retrieval_method = "semantic_scholar_search"
                        return source
            except Exception:
                pass

        if candidate_urls:
            source.source_url = candidate_urls[0]
            source.error_message = "Found URL but could not extract abstract"

    except Exception as e:
        logger.error(f"Retrieval error for {reference.id}: {e}")
        source.error_message = f"Retrieval error: {str(e)}"

    return source


def is_simple_scrapable_url(url: str) -> bool:
    """Check if URL is from a known scrapable site."""
    if not url:
        return False
    domain = urlparse(url).netloc.lower()
    simple_domains = [
        'arxiv.org', 'semanticscholar.org', 'acm.org',
        'ieee.org', 'springer.com', 'sciencedirect.com'
    ]
    return any(d in domain for d in simple_domains)


def clear_cache():
    """Clear caches."""
    _doi_cache.clear()
    _abstract_cache.clear()
    logger.info("Caches cleared")