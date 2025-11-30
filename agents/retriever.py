"""
retriever.py

Tiered retriever: API-first, browser fallback.
"""

import asyncio
from typing import List

from config import Config
from schemas import Reference, RetrievedSource
from tools import web_tools, browser_tools


async def retrieve_single(reference: Reference) -> RetrievedSource:
    """
    Retrieve source for a reference.
    Strategy: API (fast) → Browser (fallback)
    """
    title_preview = (reference.title or "Unknown")[:50]
    print(f"  Retrieving: {reference.id} - {title_preview}...")

    # 1) Fast path: API
    source = await web_tools.get_paper_content(reference)

    if source.retrieval_success:
        print(f"    ✓ Retrieved via {source.retrieval_method}")
        return source

    # 2) Fallback: Browser
    print("    Trying browser fallback...")
    try:
        browser_result = await browser_tools.find_paper_by_reference(reference)

        if browser_result.get("success"):
            source.source_url = browser_result.get("source_url")
            source.abstract = browser_result.get("abstract")
            source.full_text_snippet = browser_result.get("full_text_snippet")
            source.retrieval_success = True
            source.retrieval_method = "browser"
            print("    ✓ Retrieved via browser")
            return source

    except Exception as e:
        source.error_message = f"Browser error: {e}"
        print(f"    ✗ Error: {e}")
        return source

    source.error_message = "All retrieval methods failed"
    print("    ✗ Failed")
    return source


async def retrieve_all_async(references: List[Reference]) -> List[RetrievedSource]:
    """Retrieve sources with batched concurrency."""
    print(f"\nRetrieving {len(references)} sources...")

    sources: List[RetrievedSource] = []
    batch_size = getattr(Config, "MAX_CONCURRENT_REQUESTS", 5)

    for i in range(0, len(references), batch_size):
        batch = references[i : i + batch_size]
        results = await asyncio.gather(
            *[retrieve_single(ref) for ref in batch],
            return_exceptions=True,
        )

        for ref, result in zip(batch, results):
            if isinstance(result, Exception):
                sources.append(
                    RetrievedSource(
                        reference_id=ref.id,
                        retrieval_success=False,
                        error_message=str(result),
                    )
                )
            else:
                sources.append(result)

    successful = sum(1 for s in sources if s.retrieval_success)
    print(f"\nRetrieval: {successful}/{len(references)} successful")
    return sources


def retrieve_all(references: List[Reference]) -> List[RetrievedSource]:
    """Sync wrapper."""
    return asyncio.run(retrieve_all_async(references))