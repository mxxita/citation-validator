from typing import List, Tuple
from pathlib import Path

from config import Config
from schemas import Citation, Reference, PaperMetadata
from tools import pdf_tools


class ParserAgent:
    """Deterministic parser for extracting citations and references from PDFs."""

    def run(self, pdf_path: str | Path) -> Tuple[List[Citation], List[Reference], PaperMetadata]:
        """
        Extract citations and references from a PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (citations, references, metadata)
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"Parsing PDF: {pdf_path.name}")

        # Step 1: Extract text
        print("  - Extracting text...")
        text = pdf_tools.extract_text_from_pdf(pdf_path)

        if not text or len(text) < 100:
            raise ValueError("Failed to extract meaningful text from PDF")

        # Step 2: Extract metadata
        print("  - Extracting metadata...")
        metadata = pdf_tools.extract_metadata(pdf_path, text)

        # Step 3: Extract citations
        print("  - Extracting citations...")
        citations = pdf_tools.extract_citations(text)
        print(f"    Found {len(citations)} citations")

        # Step 4: Extract references
        print("  - Extracting references...")
        references = pdf_tools.extract_references(text)
        print(f"    Found {len(references)} references")

        # Step 5: Match citations to references
        print("  - Matching citations to references...")
        citation_mapping = pdf_tools.match_citations_to_references(citations, references)
        print(f"    Matched {len(citation_mapping)} citations")

        # Limit citations if configured
        if len(citations) > Config.MAX_CITATIONS_TO_CHECK:
            print(f"  - Limiting to {Config.MAX_CITATIONS_TO_CHECK} citations")
            citations = citations[:Config.MAX_CITATIONS_TO_CHECK]

        return citations, references, metadata


# Convenience function
def parse_pdf(pdf_path: str | Path) -> Tuple[List[Citation], List[Reference], PaperMetadata]:
    """Parse a PDF and return citations, references, and metadata."""
    parser = ParserAgent()
    return parser.run(pdf_path)