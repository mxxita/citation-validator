"""
Pydantic data models for the citation validation system.
"""

from typing import Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Represents a citation found in the paper."""

    id: str = Field(..., description="Citation marker, e.g., '[1]' or '(Smith, 2023)'")
    claim_text: str = Field(..., description="The sentence/claim containing the citation")
    page_number: int | None = Field(None, description="Page number where citation appears")
    context: str | None = Field(None, description="Surrounding context (1-2 sentences)")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "[1]",
                "claim_text": "Previous work shows that LLMs hallucinate citations 40% of the time [1].",
                "page_number": 3,
                "context": "Recent studies have examined LLM reliability. Previous work shows that LLMs hallucinate citations 40% of the time [1]. This poses challenges for academic writing."
            }
        }


class Reference(BaseModel):
    """Represents a reference from the bibliography."""

    id: str = Field(..., description="Matches Citation.id")
    title: str = Field(..., description="Paper/article title")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    year: int | None = Field(None, description="Publication year")
    doi: str | None = Field(None, description="Digital Object Identifier")
    url: str | None = Field(None, description="Direct URL if available")
    publication: str | None = Field(None, description="Journal/conference name")
    raw_text: str = Field(..., description="Original reference string from bibliography")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "[1]",
                "title": "Attention Is All You Need",
                "authors": ["Vaswani", "Shazeer", "Parmar", "Uszkoreit"],
                "year": 2017,
                "doi": "10.48550/arXiv.1706.03762",
                "publication": "NeurIPS",
                "raw_text": "[1] Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS."
            }
        }


class RetrievedSource(BaseModel):
    """Represents a source that was retrieved and scraped."""

    reference_id: str = Field(..., description="Links to Reference.id")
    source_url: str | None = Field(None, description="URL where source was found")
    abstract: str | None = Field(None, description="Paper abstract")
    full_text_snippet: str | None = Field(None, description="Relevant excerpts from full text")
    retrieval_success: bool = Field(..., description="Whether retrieval succeeded")
    retrieval_method: str | None = Field(None, description="How the source was retrieved (API, browser, etc.)")
    error_message: str | None = Field(None, description="Error message if retrieval failed")

    class Config:
        json_schema_extra = {
            "example": {
                "reference_id": "[1]",
                "source_url": "https://arxiv.org/abs/1706.03762",
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
                "full_text_snippet": None,
                "retrieval_success": True,
                "retrieval_method": "arxiv_api",
                "error_message": None
            }
        }


class ValidationResult(BaseModel):
    """Result of validating a single citation against its source."""

    citation_id: str = Field(..., description="Links to Citation.id")
    claim: str = Field(..., description="The claim being made")
    status: Literal["supported", "misrepresented", "not_found", "unverifiable"] = Field(
        ...,
        description="Validation status"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    source_excerpt: str | None = Field(None, description="What the source actually says")
    reasoning: str = Field(..., description="Explanation of the validation decision")
    suggestion: str | None = Field(None, description="How to fix if problematic")
    similarity_score: float | None = Field(None, description="Embedding similarity score")

    class Config:
        json_schema_extra = {
            "example": {
                "citation_id": "[1]",
                "claim": "Previous work shows that LLMs hallucinate citations 40% of the time",
                "status": "misrepresented",
                "confidence": 0.85,
                "source_excerpt": "Our study found hallucination rates between 15-25% in production systems.",
                "reasoning": "The source reports 15-25% hallucination rate, not 40%. The claim overstates the findings.",
                "suggestion": "Update the statistic to match the source (15-25%) or cite a different source that reports 40%.",
                "similarity_score": 0.72
            }
        }


class ReportSummary(BaseModel):
    """Summary statistics and recommendations for the validation report."""

    fulfilled: list[str] = Field(default_factory=list, description="Requirements that are fulfilled")
    missing: list[str] = Field(default_factory=list, description="Missing or problematic elements")
    desk_rejection_risk: int = Field(..., ge=1, le=5, description="Desk rejection likelihood (1-5 scale)")
    recommendations: list[str] = Field(default_factory=list, description="Actionable recommendations")

    # Statistics
    total_citations: int = Field(..., description="Total citations in paper")
    citations_checked: int = Field(..., description="Number of citations actually validated")
    supported_count: int = Field(0, description="Citations that are well-supported")
    misrepresented_count: int = Field(0, description="Citations that misrepresent sources")
    not_found_count: int = Field(0, description="Sources that could not be located")
    unverifiable_count: int = Field(0, description="Sources found but insufficient info to verify")

    class Config:
        json_schema_extra = {
            "example": {
                "fulfilled": [
                    "All references have valid DOIs",
                    "Citation format is consistent (IEEE style)",
                    "85% of citations could be verified"
                ],
                "missing": [
                    "Citation [7] could not be located online",
                    "Citation [12] links to a retracted paper",
                    "3 citations misrepresent their sources"
                ],
                "desk_rejection_risk": 2,
                "recommendations": [
                    "Verify statistics in Section 3.2 against original sources",
                    "Replace citation [7] with an accessible source",
                    "Update claim in Section 4.1 to accurately reflect source [12]"
                ],
                "total_citations": 45,
                "citations_checked": 45,
                "supported_count": 38,
                "misrepresented_count": 3,
                "not_found_count": 2,
                "unverifiable_count": 2
            }
        }


class FinalReport(BaseModel):
    """Complete validation report for a paper."""

    paper_title: str = Field(..., description="Title of the paper being validated")
    paper_authors: list[str] = Field(default_factory=list, description="Authors of the paper")
    total_citations: int = Field(..., description="Total citations found in paper")
    citations_checked: int = Field(..., description="Number of citations validated")
    results: list[ValidationResult] = Field(default_factory=list, description="Individual validation results")
    summary: ReportSummary = Field(..., description="Summary and recommendations")
    metadata: dict = Field(default_factory=dict, description="Additional metadata (timestamps, versions, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "paper_title": "Understanding LLM Hallucinations in Academic Writing",
                "paper_authors": ["Jane Doe", "John Smith"],
                "total_citations": 45,
                "citations_checked": 45,
                "results": [],  # Would contain ValidationResult objects
                "summary": {},  # Would contain ReportSummary object
                "metadata": {
                    "validation_date": "2025-11-29T10:30:00Z",
                    "validator_version": "1.0.0",
                    "processing_time_seconds": 342.5
                }
            }
        }


class PaperMetadata(BaseModel):
    """Metadata extracted from a paper."""

    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    total_pages: int | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Understanding LLM Hallucinations",
                "authors": ["Jane Doe", "John Smith", "Alice Johnson"],
                "abstract": "This paper examines the phenomenon of hallucinations in large language models...",
                "keywords": ["LLMs", "hallucinations", "AI safety"],
                "publication_year": 2024,
                "total_pages": 12
            }
        }
