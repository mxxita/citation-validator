"""
Browser tools for retrieving academic papers.
Uses Playwright for sites requiring JavaScript rendering.

CHANGES:
- Fixed ACM abstract extraction
- Added DBLP as Google Scholar alternative (no CAPTCHA)
- Improved fallback chain: DOI → DBLP → Direct URL
- Better error handling
"""

import asyncio
import re
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urljoin

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

from schemas import Reference


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def find_paper_by_reference(reference: Reference) -> Dict:
    """
    Search for a paper using browser automation.
    
    Strategy: DOI → DBLP search → Google Scholar (last resort)
    """
    query_parts = []
    if reference.title:
        query_parts.append(reference.title)
    if reference.authors:
        query_parts.append(reference.authors[0])
    
    if not query_parts:
        return {"success": False, "error": "No title or authors to search"}
    
    query = " ".join(query_parts)
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            
            # Strategy 1: DOI resolution (most reliable)
            if reference.doi:
                result = await _try_doi_resolution(page, reference.doi)
                if result["success"]:
                    await browser.close()
                    return result
            
            # Strategy 2: DBLP search (no CAPTCHA, good for CS papers)
            result = await _search_dblp(page, query)
            if result["success"]:
                await browser.close()
                return result
            
            # Strategy 3: Direct URL if provided
            if reference.url:
                result = await _try_direct_url(page, reference.url)
                if result["success"]:
                    await browser.close()
                    return result
            
            # Strategy 4: Google Scholar (often blocked)
            result = await _search_google_scholar_for_paper(page, query)
            await browser.close()
            return result
            
    except PlaywrightTimeout:
        return {"success": False, "error": "Browser timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# DOI RESOLUTION
# =============================================================================

async def _try_doi_resolution(page, doi: str) -> Dict:
    """Resolve DOI and extract abstract."""
    try:
        await page.goto(f"https://doi.org/{doi}", timeout=20000)
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(2)  # Wait for redirects and JS
        
        abstract = await _extract_abstract(page)
        url = page.url
        
        if abstract:
            return {
                "source_url": url,
                "abstract": abstract,
                "full_text_snippet": abstract[:500],
                "success": True
            }
        
        return {
            "source_url": url,
            "abstract": None,
            "success": False,
            "error": "DOI resolved but no abstract found"
        }
        
    except Exception as e:
        return {"success": False, "error": f"DOI resolution failed: {e}"}


# =============================================================================
# DBLP SEARCH (Good alternative to Google Scholar - no CAPTCHA!)
# =============================================================================

async def _search_dblp(page, query: str) -> Dict:
    """Search DBLP and navigate to paper."""
    try:
        search_url = f"https://dblp.org/search?q={quote_plus(query)}"
        await page.goto(search_url, timeout=15000)
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(1)
        
        # Get first result's link to the actual paper
        # DBLP shows external links (DOI, arXiv, etc.)
        result_entry = await page.query_selector(".entry.article, .entry.inproceedings")
        
        if not result_entry:
            return {"success": False, "error": "No results on DBLP"}
        
        # Try to find DOI or direct link
        doi_link = await result_entry.query_selector("a[href*='doi.org']")
        arxiv_link = await result_entry.query_selector("a[href*='arxiv.org']")
        
        target_url = None
        if arxiv_link:
            target_url = await arxiv_link.get_attribute("href")
        elif doi_link:
            target_url = await doi_link.get_attribute("href")
        
        if not target_url:
            # Try any external link
            external_link = await result_entry.query_selector("a.toc-link")
            if external_link:
                target_url = await external_link.get_attribute("href")
        
        if not target_url:
            return {"success": False, "error": "DBLP found result but no usable link"}
        
        # Navigate to the paper
        await page.goto(target_url, timeout=15000)
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(2)
        
        abstract = await _extract_abstract(page)
        
        return {
            "source_url": page.url,
            "abstract": abstract,
            "full_text_snippet": abstract[:500] if abstract else None,
            "success": bool(abstract)
        }
        
    except Exception as e:
        return {"success": False, "error": f"DBLP search failed: {e}"}


# =============================================================================
# DIRECT URL
# =============================================================================

async def _try_direct_url(page, url: str) -> Dict:
    """Try navigating directly to a URL."""
    try:
        await page.goto(url, timeout=15000)
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(2)
        
        abstract = await _extract_abstract(page)
        
        return {
            "source_url": page.url,
            "abstract": abstract,
            "full_text_snippet": abstract[:500] if abstract else None,
            "success": bool(abstract)
        }
    except Exception as e:
        return {"success": False, "error": f"Direct URL failed: {e}"}


# =============================================================================
# GOOGLE SCHOLAR (often blocked by CAPTCHA)
# =============================================================================

async def _search_google_scholar_for_paper(page, query: str) -> Dict:
    """Search Google Scholar - use as last resort due to CAPTCHA."""
    try:
        search_url = f"https://scholar.google.com/scholar?q={quote_plus(query)}"
        await page.goto(search_url, timeout=15000)
        await page.wait_for_load_state("domcontentloaded")
        
        # Check for CAPTCHA
        content = await page.content()
        if "captcha" in content.lower() or "unusual traffic" in content.lower():
            return {"success": False, "error": "Google Scholar CAPTCHA detected"}
        
        # Get first result link
        result_link = await page.query_selector(".gs_rt a")
        if not result_link:
            return {"success": False, "error": "No results found on Google Scholar"}
        
        href = await result_link.get_attribute("href")
        if not href:
            return {"success": False, "error": "No link in search result"}
        
        await page.goto(href, timeout=15000)
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(2)
        
        abstract = await _extract_abstract(page)
        
        return {
            "source_url": page.url,
            "abstract": abstract,
            "full_text_snippet": abstract[:500] if abstract else None,
            "success": bool(abstract)
        }
        
    except Exception as e:
        return {"success": False, "error": f"Google Scholar failed: {e}"}


# =============================================================================
# ABSTRACT EXTRACTION - IMPROVED WITH MORE SELECTORS
# =============================================================================

async def _extract_abstract(page) -> Optional[str]:
    """Extract abstract from page with site-specific handling."""
    
    url = page.url.lower()
    
    # Site-specific extraction
    if "arxiv.org" in url:
        return await _extract_arxiv(page)
    elif "acm.org" in url:
        return await _extract_acm(page)
    elif "ieee.org" in url:
        return await _extract_ieee(page)
    elif "springer.com" in url or "nature.com" in url:
        return await _extract_springer_nature(page)
    elif "semanticscholar.org" in url:
        return await _extract_semantic_scholar(page)
    elif "sciencedirect.com" in url:
        return await _extract_sciencedirect(page)
    elif "neurips.cc" in url or "nips.cc" in url:
        return await _extract_neurips(page)
    elif "openreview.net" in url:
        return await _extract_openreview(page)
    elif "mlr.press" in url:
        return await _extract_mlr(page)
    elif "aclanthology.org" in url or "aclweb.org" in url:
        return await _extract_acl(page)
    else:
        return await _extract_generic(page)


async def _extract_neurips(page) -> Optional[str]:
    """Extract from NeurIPS proceedings."""
    selectors = [
        "div.abstract",
        "p.abstract", 
        "#abstract",
        "div#abstractSection",
        ".paper-abstract",
    ]
    for sel in selectors:
        el = await page.query_selector(sel)
        if el:
            text = await el.inner_text()
            if len(text.strip()) > 50:
                return _clean_abstract(text)
    
    # NeurIPS sometimes has abstract after "Abstract" heading
    headings = await page.query_selector_all("h4, h3, h2")
    for h in headings:
        text = await h.inner_text()
        if "abstract" in text.lower():
            next_p = await h.evaluate("el => el.nextElementSibling?.innerText")
            if next_p and len(next_p.strip()) > 50:
                return _clean_abstract(next_p)
    
    return await _extract_generic(page)


async def _extract_openreview(page) -> Optional[str]:
    """Extract from OpenReview."""
    selectors = [
        "div.note-content-value",
        "span.note-content-value",
        "div[class*='abstract']",
        ".note-content",
    ]
    for sel in selectors:
        el = await page.query_selector(sel)
        if el:
            text = await el.inner_text()
            if len(text.strip()) > 50:
                return _clean_abstract(text)
    
    # Try looking for "Abstract" field specifically
    fields = await page.query_selector_all(".note-content-field")
    for field in fields:
        label = await field.query_selector(".note-content-field-name")
        if label:
            label_text = await label.inner_text()
            if "abstract" in label_text.lower():
                value = await field.query_selector(".note-content-value")
                if value:
                    text = await value.inner_text()
                    return _clean_abstract(text)
    
    return await _extract_generic(page)


async def _extract_mlr(page) -> Optional[str]:
    """Extract from PMLR (ICML, AISTATS, etc.)."""
    selectors = [
        "#abstract",
        ".abstract",
        "div.abstract",
    ]
    for sel in selectors:
        el = await page.query_selector(sel)
        if el:
            text = await el.inner_text()
            if len(text.strip()) > 50:
                return _clean_abstract(text)
    return await _extract_generic(page)


async def _extract_acl(page) -> Optional[str]:
    """Extract from ACL Anthology."""
    selectors = [
        ".acl-abstract",
        "div.card-body span.d-block",
        ".abstract",
        "#abstract",
    ]
    for sel in selectors:
        el = await page.query_selector(sel)
        if el:
            text = await el.inner_text()
            if len(text.strip()) > 50:
                return _clean_abstract(text)
    return await _extract_generic(page)


async def _extract_arxiv(page) -> Optional[str]:
    """Extract from arXiv."""
    selectors = [
        "blockquote.abstract",
        ".abstract",
        "#abs",
    ]
    for sel in selectors:
        el = await page.query_selector(sel)
        if el:
            text = await el.inner_text()
            return _clean_abstract(text)
    return None


async def _extract_acm(page) -> Optional[str]:
    """Extract from ACM Digital Library - FIXED."""
    selectors = [
        # New ACM layout (2023+)
        "div.abstractSection.abstractInFull",
        "div.abstractSection",
        "section.abstract div.abstractInFull",
        "section.abstract",
        # Older layout
        "div.article__abstract",
        "div#abstract",
        # Role-based
        "div[role='paragraph']",
        # Class contains
        "[class*='abstract']",
    ]
    
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                text = await el.inner_text()
                if text and len(text.strip()) > 100:
                    return _clean_abstract(text)
        except Exception:
            continue
    
    # Try meta description as fallback
    meta = await page.query_selector("meta[name='description']")
    if meta:
        content = await meta.get_attribute("content")
        if content and len(content) > 100:
            return content
    
    return None


async def _extract_ieee(page) -> Optional[str]:
    """Extract from IEEE Xplore."""
    selectors = [
        "div.abstract-text",
        "div[class*='abstract']",
        "xpl-abstract",
    ]
    for sel in selectors:
        el = await page.query_selector(sel)
        if el:
            text = await el.inner_text()
            if len(text.strip()) > 100:
                return _clean_abstract(text)
    return None


async def _extract_springer_nature(page) -> Optional[str]:
    """Extract from Springer/Nature."""
    selectors = [
        "#Abs1-content",
        "#Abs1",
        "div.c-article-section__content",
        "section[data-title='Abstract'] p",
        "#abstract-content",
        ".article__abstract",
    ]
    for sel in selectors:
        el = await page.query_selector(sel)
        if el:
            text = await el.inner_text()
            if len(text.strip()) > 50:
                return _clean_abstract(text)
    return None


async def _extract_semantic_scholar(page) -> Optional[str]:
    """Extract from Semantic Scholar."""
    selectors = [
        "div[data-test-id='abstract-text']",
        "div.text-truncator",
        "div.abstract__text",
        "meta[name='description']",
    ]
    for sel in selectors:
        try:
            if sel.startswith("meta"):
                el = await page.query_selector(sel)
                if el:
                    content = await el.get_attribute("content")
                    if content and len(content) > 100:
                        return content
            else:
                el = await page.query_selector(sel)
                if el:
                    text = await el.inner_text()
                    if len(text.strip()) > 100:
                        return _clean_abstract(text)
        except Exception:
            continue
    return None


async def _extract_sciencedirect(page) -> Optional[str]:
    """Extract from ScienceDirect."""
    selectors = [
        "div.abstract.author",
        "div#abstracts",
        "div[class*='abstract']",
    ]
    for sel in selectors:
        el = await page.query_selector(sel)
        if el:
            text = await el.inner_text()
            if len(text.strip()) > 100:
                return _clean_abstract(text)
    return None


async def _extract_generic(page) -> Optional[str]:
    """Generic extraction for unknown sites."""
    # Try common selectors
    selectors = [
        "#abstract",
        ".abstract",
        "[class*='abstract' i]",
        "[id*='abstract' i]",
        "section.abstract",
    ]
    
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                text = await el.inner_text()
                if text and 100 < len(text.strip()) < 5000:
                    return _clean_abstract(text)
        except Exception:
            continue
    
    # Meta tags as last resort
    meta_selectors = [
        "meta[name='description']",
        "meta[property='og:description']",
        "meta[name='abstract']",
        "meta[name='DC.description']",
    ]
    
    for sel in meta_selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                content = await el.get_attribute("content")
                if content and 100 < len(content) < 3000:
                    return content
        except Exception:
            continue
    
    return None


def _clean_abstract(text: str) -> str:
    """Clean extracted abstract text."""
    if not text:
        return ""
    
    text = text.strip()
    
    # Remove common prefixes
    prefixes = ["Abstract", "ABSTRACT", "Summary", "SUMMARY", "TLDR", "TL;DR"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip(":.-– \n")
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    return text


# =============================================================================
# PUBLIC HELPER FUNCTIONS
# =============================================================================

async def search_google_scholar(query: str, max_results: int = 5) -> List[Dict]:
    """Search Google Scholar - may be blocked by CAPTCHA."""
    results = []
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            )
            page = await context.new_page()
            
            await page.goto(f"https://scholar.google.com/scholar?q={quote_plus(query)}", timeout=15000)
            await page.wait_for_load_state("domcontentloaded")
            
            content = await page.content()
            if "captcha" in content.lower() or "unusual traffic" in content.lower():
                await browser.close()
                return []
            
            items = await page.query_selector_all(".gs_r.gs_or.gs_scl")
            
            for item in items[:max_results]:
                try:
                    title_el = await item.query_selector(".gs_rt a")
                    title = await title_el.inner_text() if title_el else None
                    url = await title_el.get_attribute("href") if title_el else None
                    
                    authors_el = await item.query_selector(".gs_a")
                    authors_text = await authors_el.inner_text() if authors_el else ""
                    
                    authors = None
                    year = None
                    if authors_text:
                        parts = authors_text.split(" - ")
                        if parts:
                            authors = parts[0].strip()
                        year_match = re.search(r'\b(19|20)\d{2}\b', authors_text)
                        if year_match:
                            year = int(year_match.group())
                    
                    snippet_el = await item.query_selector(".gs_rs")
                    snippet = await snippet_el.inner_text() if snippet_el else None
                    
                    results.append({
                        "title": title,
                        "url": url,
                        "authors": authors,
                        "year": year,
                        "snippet": snippet
                    })
                except Exception:
                    continue
            
            await browser.close()
            
    except Exception as e:
        print(f"Google Scholar error: {e}")
    
    return results


async def search_dblp(query: str, max_results: int = 5) -> List[Dict]:
    """Search DBLP - reliable, no CAPTCHA."""
    results = []
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            await page.goto(f"https://dblp.org/search?q={quote_plus(query)}", timeout=15000)
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(1)
            
            entries = await page.query_selector_all(".entry.article, .entry.inproceedings, .entry.conf")
            
            for entry in entries[:max_results]:
                try:
                    # Title
                    title_el = await entry.query_selector(".title")
                    title = await title_el.inner_text() if title_el else None
                    
                    # Authors
                    author_els = await entry.query_selector_all("[itemprop='author'] span")
                    authors = []
                    for ael in author_els:
                        name = await ael.inner_text()
                        if name:
                            authors.append(name.strip())
                    
                    # Year
                    year_el = await entry.query_selector("[itemprop='datePublished']")
                    year = None
                    if year_el:
                        year_text = await year_el.inner_text()
                        if year_text:
                            year = int(year_text.strip())
                    
                    # Links
                    doi_el = await entry.query_selector("a[href*='doi.org']")
                    arxiv_el = await entry.query_selector("a[href*='arxiv.org']")
                    
                    url = None
                    if arxiv_el:
                        url = await arxiv_el.get_attribute("href")
                    elif doi_el:
                        url = await doi_el.get_attribute("href")
                    
                    results.append({
                        "title": title.strip() if title else None,
                        "authors": authors,
                        "year": year,
                        "url": url,
                    })
                except Exception:
                    continue
            
            await browser.close()
            
    except Exception as e:
        print(f"DBLP search error: {e}")
    
    return results


async def navigate_to_paper(url: str) -> Dict:
    """Navigate to paper URL and extract content."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            )
            page = await context.new_page()
            
            await page.goto(url, timeout=15000)
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(2)
            
            abstract = await _extract_abstract(page)
            
            # Look for PDF link
            pdf_url = None
            pdf_selectors = ["a[href*='.pdf']", "a:has-text('PDF')", "a:has-text('Download')"]
            for sel in pdf_selectors:
                try:
                    pdf_link = await page.query_selector(sel)
                    if pdf_link:
                        pdf_url = await pdf_link.get_attribute("href")
                        if pdf_url and not pdf_url.startswith("http"):
                            pdf_url = urljoin(url, pdf_url)
                        break
                except Exception:
                    continue
            
            await browser.close()
            
            return {
                "abstract": abstract,
                "pdf_url": pdf_url,
                "full_text_available": bool(abstract and len(abstract) > 500),
                "success": bool(abstract)
            }
            
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# SYNC WRAPPERS
# =============================================================================

def find_paper_sync(reference: Reference) -> Dict:
    return asyncio.run(find_paper_by_reference(reference))

def search_scholar_sync(query: str, max_results: int = 5) -> List[Dict]:
    return asyncio.run(search_google_scholar(query, max_results))

def search_dblp_sync(query: str, max_results: int = 5) -> List[Dict]:
    return asyncio.run(search_dblp(query, max_results))

def navigate_sync(url: str) -> Dict:
    return asyncio.run(navigate_to_paper(url))