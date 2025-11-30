"""
PDF parsing tools for citation extraction.

FIXES:
- Better title extraction (skips copyright, headers, arXiv IDs)
- Improved reference section extraction
- Better author parsing
- Removed duplicate functions
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import fitz
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from config import Config
from schemas import Citation, Reference, PaperMetadata


ABBREVIATIONS = {
    'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr',
    'fig', 'figs', 'eq', 'eqs', 'ref', 'refs', 'sec', 'secs',
    'vol', 'no', 'pp', 'ed', 'eds', 'et al', 'ie', 'eg', 'cf', 'vs',
    'inc', 'corp', 'ltd', 'co', 'dept', 'st', 'ave',
}

# Patterns to skip when looking for title
SKIP_PATTERNS = [
    r'^arXiv:\d+\.\d+',  # arXiv ID
    r'^https?://',  # URLs
    r'^www\.',
    r'^\d+$',  # Just numbers
    r'^page\s+\d+',
    r'^proceedings\s+of',
    r'^published\s+in',
    r'^submitted\s+to',
    r'^copyright',
    r'^©',
    r'^\(c\)',
    r'^all\s+rights\s+reserved',
    r'^permission\s+to',
    r'^provided\s+proper',  # "Provided proper attribution..."
    r'^google\s+here?by',
    r'^license',
    r'^creative\s+commons',
    r'^abstract\s*$',
    r'^keywords?\s*:',
    r'^introduction\s*$',
    r'^\d+\.\s+\w',  # Section numbers like "1. Introduction"
    r'^conference\s+on',
    r'^\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',  # Dates
]


def _get_llm():
    """Get LLM instance for claim extraction."""
    llm_config = Config.get_llm_config()

    if llm_config["provider"] == "openai":
        return ChatOpenAI(
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            temperature=0
        )
    elif llm_config["provider"] == "anthropic":
        return ChatAnthropic(
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            temperature=0
        )
    else:
        raise ValueError(f"Unknown LLM provider: {llm_config['provider']}")


def clean_text(text: str) -> str:
    """Remove headers, footers, page markers from extracted PDF text."""
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line_stripped = line.strip()

        if not line_stripped:
            continue

        if line_stripped.startswith('--- PAGE'):
            continue

        if re.match(r'arXiv:\d+\.\d+v?\d*\s*\[[\w.]+\]\s*\d+\s+\w+\s+\d{4}', line_stripped, re.IGNORECASE):
            continue

        if len(line_stripped) < 5:
            continue

        footer_patterns = [
            r'^\d+$',
            r'^page\s+\d+',
            r'^\d+\s+of\s+\d+$',
            r'^proceedings of',
            r'^conference on',
            r'^\d{4}\s+(ieee|acm|springer)',
            r'^submitted to',
            r'^preprint',
        ]

        if any(re.match(pattern, line_stripped, re.IGNORECASE) for pattern in footer_patterns):
            continue

        if re.match(r'^(https?://|www\.|.*@.*\.)', line_stripped):
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract full text from a PDF file."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        text_content = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            text_content.append(f"\n--- PAGE {page_num} ---\n{text}")

        doc.close()
        return "\n".join(text_content)

    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")


def extract_metadata(pdf_path: str | Path, text: Optional[str] = None) -> PaperMetadata:
    """Extract metadata from PDF (title, authors, etc.)."""
    pdf_path = Path(pdf_path)
    metadata = PaperMetadata()

    try:
        doc = fitz.open(pdf_path)

        # Try PDF metadata first
        pdf_meta = doc.metadata
        if pdf_meta:
            title = pdf_meta.get("title")
            # Validate title - PDF metadata is often garbage
            if title and len(title) > 10 and not _is_garbage_title(title):
                metadata.title = title

            if pdf_meta.get("author"):
                authors_str = pdf_meta["author"]
                authors = re.split(r'[,;]|\sand\s', authors_str)
                metadata.authors = [a.strip() for a in authors if a.strip()]

        metadata.total_pages = len(doc)
        doc.close()

        # If no good title from metadata, extract from text
        if not metadata.title and text:
            metadata.title = _extract_title_from_text(text)
        
        # If no authors from metadata, extract from text
        if not metadata.authors and text:
            metadata.authors = _extract_authors_from_text(text)

        # Extract abstract
        if text:
            abstract_match = re.search(
                r'abstract[:\s]+(.*?)(?=\n\s*\n|\n[A-Z][a-z]+:|\n\d+\.?\s+[A-Z])',
                text,
                re.IGNORECASE | re.DOTALL
            )
            if abstract_match:
                metadata.abstract = abstract_match.group(1).strip()[:1000]

    except Exception as e:
        print(f"Warning: Could not extract metadata: {e}")

    return metadata


def _is_garbage_title(title: str) -> bool:
    """Check if a title looks like garbage/metadata."""
    title_lower = title.lower()
    garbage_indicators = [
        'untitled',
        'microsoft word',
        'document',
        '.pdf',
        '.doc',
        'copyright',
        'permission',
        'arxiv',
        'proceedings',
    ]
    return any(ind in title_lower for ind in garbage_indicators)


def _extract_title_from_text(text: str) -> Optional[str]:
    """
    Extract title from first page text.
    
    IMPROVED: Skips copyright notices, arXiv headers, etc.
    """
    # Get first page only
    first_page = text.split("--- PAGE 2 ---")[0] if "--- PAGE 2 ---" in text else text[:3000]
    
    # Remove page marker
    first_page = re.sub(r'---\s*PAGE\s*\d+\s*---', '', first_page)
    
    lines = [line.strip() for line in first_page.split('\n') if line.strip()]
    
    candidates = []
    
    for line in lines[:20]:  # Check first 20 lines
        # Skip short lines
        if len(line) < 15:
            continue
        
        # Skip lines matching garbage patterns
        if any(re.match(pattern, line, re.IGNORECASE) for pattern in SKIP_PATTERNS):
            continue
        
        # Skip lines that are all uppercase and very short (headers)
        if line.isupper() and len(line) < 40:
            continue
        
        # Skip lines that look like author lists (multiple commas, institutional patterns)
        if line.count(',') > 3:
            continue
        if re.search(r'university|institute|department|@|\.edu|\.org', line, re.IGNORECASE):
            continue
        
        # Skip lines that start with lowercase (likely continuation)
        if line[0].islower():
            continue
        
        # This looks like a potential title
        candidates.append(line)
    
    if candidates:
        # Return the first good candidate
        # Titles are often the longest line in the first few candidates
        # But we'll just take the first one that passed all filters
        return candidates[0]
    
    return None


def _extract_authors_from_text(text: str) -> List[str]:
    """Extract authors from the first page."""
    first_page = text.split("--- PAGE 2 ---")[0] if "--- PAGE 2 ---" in text else text[:3000]
    
    lines = first_page.split('\n')
    authors = []
    
    # Strategy 1: Look for lines with multiple names separated by superscripts/symbols
    # Common in NeurIPS/ICML papers: "Ashish Vaswani∗ Noam Shazeer∗ Niki Parmar∗"
    for line in lines[:30]:
        line = line.strip()
        
        if len(line) < 10:
            continue
        
        # Skip title-like lines (usually longer, no special symbols)
        if len(line) > 100:
            continue
            
        # Skip lines that look like affiliations
        if re.search(r'university|institute|department|google|facebook|microsoft|@|\.edu|\.org|\.com', line, re.IGNORECASE):
            continue
        
        # Skip lines starting with common non-author patterns
        if re.match(r'^(abstract|introduction|\d+\.|keywords|arxiv|http|www)', line, re.IGNORECASE):
            continue
        
        # Look for author patterns:
        # - Multiple capitalized words with symbols between them
        # - Format: "FirstName LastName∗" or "FirstName LastName1,2"
        
        # Check if line has multiple name-like patterns
        # Name pattern: Capital letter followed by lowercase, possibly with symbols after
        name_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z\']+)+)[∗†‡§¶\d,\s]*'
        potential_names = re.findall(name_pattern, line)
        
        if len(potential_names) >= 2:
            # This looks like an author line
            for name in potential_names:
                name = name.strip()
                if 5 < len(name) < 40:
                    authors.append(name)
            if authors:
                break  # Found author line, stop searching
    
    # Strategy 2: If strategy 1 failed, look for "Name and Name" patterns
    if not authors:
        for line in lines[:30]:
            line = line.strip()
            if ' and ' in line and len(line) < 100:
                # Check if it looks like names
                parts = re.split(r'\s+and\s+|,\s*', line)
                for part in parts:
                    part = part.strip()
                    # Check if it looks like a name (2-3 capitalized words)
                    words = part.split()
                    if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                        if not re.search(r'university|institute|department', part, re.IGNORECASE):
                            authors.append(part)
    
    # Strategy 3: Look for lines that are just names (common in some formats)
    if not authors:
        for line in lines[:20]:
            line = line.strip()
            # Line with 2-4 words, all starting with capitals, reasonable length
            words = line.split()
            if 2 <= len(words) <= 4 and 10 < len(line) < 35:
                if all(w[0].isupper() for w in words):
                    if not re.search(r'abstract|introduction|university|arxiv|figure|table', line, re.IGNORECASE):
                        authors.append(line)
    
    # Deduplicate while preserving order
    seen = set()
    unique_authors = []
    for author in authors:
        if author.lower() not in seen:
            seen.add(author.lower())
            unique_authors.append(author)
    
    return unique_authors[:10]


def extract_claim_with_llm(text: str, citation_position: int, citation_id: str, llm=None) -> str:
    """Use LLM to extract the actual claim/sentence containing a citation."""
    if llm is None:
        llm = _get_llm()

    window = 600
    context_start = max(0, citation_position - window)
    context_end = min(len(text), citation_position + window)
    context = text[context_start:context_end]

    prompt = f"""Extract the complete sentence or claim that uses the citation {citation_id}.

Context:
{context}

Instructions:
1. Find the sentence that contains {citation_id}
2. Return ONLY that complete sentence
3. Do NOT include page numbers, headers, footers, or metadata
4. Keep the citation marker {citation_id} in the sentence

Return only the extracted sentence/claim, nothing else."""

    try:
        response = llm.invoke(prompt)
        claim = response.content.strip()
        claim = re.sub(r'^(The claim is:|The sentence is:|Extracted claim:)\s*["\']?', '', claim, flags=re.IGNORECASE)
        claim = re.sub(r'^["\']|["\']$', '', claim)

        if len(claim) < 20:
            return _extract_claim_sentence(text, citation_position)

        return claim

    except Exception as e:
        print(f"LLM claim extraction failed: {e}. Falling back to regex.")
        return _extract_claim_sentence(text, citation_position)


def extract_citations(text: str, use_llm: bool = False) -> List[Citation]:
    """Extract citations from text."""
    print("    Cleaning text...")
    cleaned_text = clean_text(text)

    llm = _get_llm() if use_llm else None

    citations = []
    seen = set()

    # Pattern for numeric citations [1], [1-3], [1,2], [1, 2, 3]
    numeric_pattern = r'\[(\d+(?:[-–,]\s*\d+)*)\]'

    # Pattern for author-year citations
    author_year_pattern = r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?(?:\s+and\s+[A-Z][a-z]+)?),\s*(\d{4}[a-z]?)\)'

    print("    Extracting claims with LLM..." if use_llm else "    Extracting claims with regex...")
    
    for match in re.finditer(numeric_pattern, cleaned_text):
        citation_id = match.group(0)
        position = match.start()
        page_number = _estimate_page_number(text, position)

        key = (citation_id, page_number)
        if key in seen:
            continue
        seen.add(key)

        if use_llm and llm:
            claim_text = extract_claim_with_llm(cleaned_text, position, citation_id, llm)
        else:
            claim_text = _extract_claim_sentence(cleaned_text, position)

        context = cleaned_text[max(0, position - 150):min(len(cleaned_text), position + 150)].strip()

        citations.append(Citation(
            id=citation_id,
            claim_text=claim_text,
            page_number=page_number,
            context=context
        ))

    for match in re.finditer(author_year_pattern, cleaned_text):
        author = match.group(1)
        year = match.group(2)
        citation_id = f"({author}, {year})"
        position = match.start()
        page_number = _estimate_page_number(text, position)

        key = (citation_id, page_number)
        if key in seen:
            continue
        seen.add(key)

        if use_llm and llm:
            claim_text = extract_claim_with_llm(cleaned_text, position, citation_id, llm)
        else:
            claim_text = _extract_claim_sentence(cleaned_text, position)

        context = cleaned_text[max(0, position - 150):min(len(cleaned_text), position + 150)].strip()

        citations.append(Citation(
            id=citation_id,
            claim_text=claim_text,
            page_number=page_number,
            context=context
        ))

    return citations


def _extract_claim_sentence(text: str, position: int, window: int = 400) -> str:
    """Extract the sentence containing a citation with proper boundary detection."""
    before_text = text[max(0, position - window):position]
    after_text = text[position:min(len(text), position + window)]
    
    # Find sentence start
    sentence_start = 0
    for match in re.finditer(r'\.\s+', before_text):
        pre_period = before_text[max(0, match.start() - 10):match.start()].lower()
        is_abbrev = any(pre_period.endswith(abbr) for abbr in ABBREVIATIONS)
        is_decimal = match.start() > 0 and before_text[match.start() - 1].isdigit()
        
        if not is_abbrev and not is_decimal:
            sentence_start = match.end()
    
    # Find sentence end
    sentence_end = len(after_text)
    for match in re.finditer(r'\.\s+', after_text):
        pre_period = after_text[max(0, match.start() - 10):match.start()].lower()
        is_abbrev = any(pre_period.endswith(abbr) for abbr in ABBREVIATIONS)
        is_decimal = match.start() > 0 and after_text[match.start() - 1].isdigit()
        
        if not is_abbrev and not is_decimal:
            sentence_end = match.start() + 1
            break
    
    claim = (before_text[sentence_start:] + after_text[:sentence_end]).strip()
    claim = re.sub(r'\s+', ' ', claim)
    
    return claim


def extract_references(text: str) -> List[Reference]:
    """Extract references from the References/Bibliography section."""
    references = []
    
    ref_section = _extract_references_section(text)
    if not ref_section:
        print("    Warning: Could not find references section")
        return references
    
    # Clean up the section
    ref_section = re.sub(r'---\s*PAGE\s*\d+\s*---', ' ', ref_section)
    
    # Split by reference numbers [1], [2], etc.
    # But handle multi-line references properly
    pattern = r'\[(\d+)\]'
    matches = list(re.finditer(pattern, ref_section))
    
    if not matches:
        print("    Warning: No numbered references found")
        return references
    
    for i, match in enumerate(matches):
        ref_num = int(match.group(1))
        start = match.end()
        
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(ref_section)
        
        entry_text = ref_section[start:end].strip()
        
        # Normalize whitespace
        entry_text = re.sub(r'\s+', ' ', entry_text)
        
        # Skip garbage entries
        if len(entry_text) < 20:
            continue
        
        ref = _parse_reference_entry(entry_text, ref_num)
        if ref:
            ref.id = f"[{ref_num}]"
            references.append(ref)
    
    return references


def _extract_references_section(text: str) -> Optional[str]:
    """Extract the references section from text."""
    # Find start of references
    patterns = [
        r'\n\s*REFERENCES\s*\n',
        r'\n\s*References\s*\n',
        r'\n\s*BIBLIOGRAPHY\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*Works\s+Cited\s*\n',
        r'\nReferences\s*\n',
        r'\n\d+\.?\s*References\s*\n',  # "8. References" or "8 References"
    ]

    start_pos = None
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_pos = match.end()
            break

    if start_pos is None:
        return None

    # Find end markers
    end_markers = [
        r'\n\s*APPENDIX',
        r'\n\s*Appendix',
        r'\n\s*A\.\s+\w',  # "A. Supplementary"
        r'\n\s*ACKNOWLEDGMENT',
        r'\n\s*Acknowledgment',
        r'\n\s*SUPPLEMENTARY',
        r'\n\s*Supplementary',
    ]
    
    end_pos = len(text)
    for end_marker in end_markers:
        end_match = re.search(end_marker, text[start_pos:], re.IGNORECASE)
        if end_match:
            end_pos = start_pos + end_match.start()
            break

    return text[start_pos:end_pos]


def _parse_reference_entry(entry: str, index: int) -> Optional[Reference]:
    """Parse a single reference entry into a Reference object."""
    ref = Reference(
        id=f"[{index}]",
        raw_text=entry,
        title="",
        authors=[]
    )
    
    # Extract year (4 digits)
    year_match = re.search(r'[\(\s,]*((?:19|20)\d{2})[a-z]?[\)\s,\.]', entry)
    if year_match:
        ref.year = int(year_match.group(1))
    
    # Extract DOI
    doi_match = re.search(r'(?:doi[:\s]*|https?://doi\.org/)(10\.\d{4,}/[^\s,\]]+)', entry, re.IGNORECASE)
    if doi_match:
        ref.doi = doi_match.group(1).rstrip('.')
    
    # Extract arXiv ID - store in URL if no other URL
    arxiv_match = re.search(r'arXiv[:\s]*(\d+\.\d+)', entry, re.IGNORECASE)
    if arxiv_match and not ref.url:
        ref.url = f"https://arxiv.org/abs/{arxiv_match.group(1)}"
    
    # Extract URL
    url_match = re.search(r'(https?://[^\s,\)\]]+)', entry)
    if url_match:
        url = url_match.group(1).rstrip('.')
        # Don't store DOI URLs separately
        if 'doi.org' not in url:
            ref.url = url
    
    # Parse authors and title
    # Common format: "Authors. Title. Venue, year."
    # Or: "Authors (year). Title. Venue."
    
    # Try to split on periods
    parts = re.split(r'\.\s+', entry, maxsplit=3)
    
    if len(parts) >= 2:
        # First part is usually authors
        authors_part = parts[0]
        
        # Remove year if it's at the end of authors part
        authors_part = re.sub(r'\s*\(\d{4}\)\s*$', '', authors_part)
        
        # Split authors
        author_names = re.split(r',\s*(?:and\s+)?|\s+and\s+', authors_part)
        ref.authors = [
            name.strip() 
            for name in author_names 
            if name.strip() and len(name.strip()) > 2 and not name.strip().isdigit()
        ][:8]  # Limit to 8 authors
        
        # Second part is usually title
        if len(parts) >= 2:
            title = parts[1].strip()
            # Clean up title
            title = re.sub(r'^\d+\.\s*', '', title)  # Remove leading numbers
            title = re.sub(r'\s*\(\d{4}\)\s*', '', title)  # Remove year
            # Remove venue info that got concatenated with title
            title = re.sub(r'\s*In\s+(Advances|Proceedings|Conference|International|Annual|IEEE|ACM|AAAI|IJCAI|ICML|NeurIPS|NIPS|ICLR|CVPR|ICCV|ECCV|ACL|EMNLP|NAACL|COLING).*$', '', title, flags=re.IGNORECASE)
            title = re.sub(r'\s*,\s*\d{4}\s*\.?\s*$', '', title)  # Remove trailing year
            if len(title) > 10:
                ref.title = title
    
    # If no title found, try quoted text
    if not ref.title:
        title_match = re.search(r'["""]([^"""]+)["""]', entry)
        if title_match:
            ref.title = title_match.group(1).strip()
    
    # Last resort: use a chunk of the entry as title
    if not ref.title and len(entry) > 30:
        # Skip authors part (before first period) and take next chunk
        first_period = entry.find('. ')
        if first_period > 0:
            rest = entry[first_period + 2:first_period + 150]
            ref.title = rest.split('.')[0].strip()
    
    return ref if (ref.title or ref.doi or ref.arxiv_id) else None


def match_citations_to_references(
    citations: List[Citation],
    references: List[Reference]
) -> Dict[str, Reference]:
    """Match citation IDs to their full reference details."""
    citation_to_ref = {}

    # Build a lookup by reference ID
    ref_lookup = {ref.id: ref for ref in references}

    for citation in citations:
        if citation.id.startswith('[') and citation.id.endswith(']'):
            # Handle single and range citations
            # [1], [1-3], [1,2,3], [1, 2]
            numbers = re.findall(r'\d+', citation.id)
            
            for num in numbers:
                num_id = f"[{num}]"
                if num_id in ref_lookup:
                    citation_to_ref[citation.id] = ref_lookup[num_id]
                    break  # Take first match for ranges

        else:
            # Author-year citations
            match = re.search(r'\(([^,]+),\s*(\d{4})', citation.id)
            if match:
                author_lastname = match.group(1).replace('et al.', '').replace('and', '').strip()
                year = int(match.group(2))

                for ref in references:
                    if ref.year == year:
                        if any(author_lastname.lower() in author.lower() for author in ref.authors):
                            citation_to_ref[citation.id] = ref
                            break

    return citation_to_ref


def _estimate_page_number(text: str, position: int) -> Optional[int]:
    """Estimate page number from position in text."""
    text_before = text[:position]
    page_matches = list(re.finditer(r'--- PAGE (\d+) ---', text_before))

    if page_matches:
        last_page_match = page_matches[-1]
        return int(last_page_match.group(1))

    return None


def preview_citations(pdf_path: str | Path, max_count: int = 10) -> None:
    """Preview citations extracted from a PDF."""
    text = extract_text_from_pdf(pdf_path)
    citations = extract_citations(text)

    print(f"\nFound {len(citations)} citations:\n")
    for i, citation in enumerate(citations[:max_count], 1):
        print(f"{i}. {citation.id} (Page {citation.page_number})")
        print(f"   Claim: {citation.claim_text[:100]}...")
        print()


def preview_references(pdf_path: str | Path, max_count: int = 10) -> None:
    """Preview references extracted from a PDF."""
    text = extract_text_from_pdf(pdf_path)
    references = extract_references(text)

    print(f"\nFound {len(references)} references:\n")
    for i, ref in enumerate(references[:max_count], 1):
        print(f"{i}. {ref.id}")
        print(f"   Title: {ref.title}")
        print(f"   Authors: {', '.join(ref.authors[:3])}")
        print(f"   Year: {ref.year}")
        print(f"   DOI: {ref.doi}")
        print()