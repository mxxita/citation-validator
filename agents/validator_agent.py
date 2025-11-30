import json
from typing import List, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from config import Config
from schemas import Citation, RetrievedSource, ValidationResult
from tools import embedding_tools
from prompts.validation_prompts import (
    VALIDATOR_SYSTEM_PROMPT,
    VALIDATOR_VALIDATE_CITATION_PROMPT,
)


def _create_llm():
    """Create LLM based on config."""
    llm_config = Config.get_llm_config()
    
    if llm_config["provider"] == "anthropic":
        return ChatAnthropic(
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            temperature=0,
        )
    
    return ChatOpenAI(
        model=llm_config["model"],
        api_key=llm_config["api_key"],
        temperature=0,
    )


LLM = _create_llm()


def _llm_validate(
    citation_id: str,
    claim: str,
    source_content: str,
    similarity_score: Optional[float],
) -> ValidationResult:
    
    prompt = VALIDATOR_VALIDATE_CITATION_PROMPT.format(
        citation_id=citation_id,
        claim=claim,
        source_content=source_content,
    )

    messages = [
        SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = LLM.invoke(messages)
        content = response.content
        
        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        data = json.loads(content.strip())

        return ValidationResult(
            citation_id=citation_id,
            claim=claim,
            status=data.get("status", "unverifiable"),
            confidence=float(data.get("confidence", 0.5)),
            source_excerpt=data.get("source_excerpt", source_content[:300]),
            reasoning=data.get("reasoning", ""),
            suggestion=data.get("suggestion"),
            similarity_score=similarity_score,
        )

    except json.JSONDecodeError as e:
        return ValidationResult(
            citation_id=citation_id,
            claim=claim,
            status="unverifiable",
            confidence=0.0,
            source_excerpt=source_content[:300],
            reasoning=f"Failed to parse LLM response: {e}",
            similarity_score=similarity_score,
        )
    except Exception as e:
        return ValidationResult(
            citation_id=citation_id,
            claim=claim,
            status="unverifiable",
            confidence=0.0,
            source_excerpt=source_content[:300],
            reasoning=f"Validation error: {e}",
            similarity_score=similarity_score,
        )


def validate_single(citation: Citation, source: RetrievedSource) -> ValidationResult:
    """Validate a citation against its retrieved source."""
    
    # No source → not_found
    if not source.retrieval_success:
        return ValidationResult(
            citation_id=citation.id,
            claim=citation.claim_text,
            status="not_found",
            confidence=1.0,
            reasoning="Source could not be retrieved.",
            suggestion="Verify the reference is correct and accessible.",
        )

    # Get source text (prefer full_text_snippet, fallback to abstract)
    source_text = source.full_text_snippet or source.abstract
    if not source_text:
        return ValidationResult(
            citation_id=citation.id,
            claim=citation.claim_text,
            status="not_found",
            confidence=1.0,
            reasoning="Retrieved source has no content.",
            suggestion="Try retrieving full text instead of abstract.",
        )

    # Find relevant passages via embeddings
    chunks = embedding_tools.chunk_text(source_text, chunk_size=300)
    top_passages = embedding_tools.find_relevant_passage(
        claim=citation.claim_text,
        source_chunks=chunks,
        top_k=3,
        min_similarity=0.5,
    )

    if top_passages:
        source_excerpt, similarity_score = top_passages[0]
    else:
        source_excerpt = source_text[:500]
        similarity_score = 0.0

    # LLM judgment
    return _llm_validate(
        citation_id=citation.id,
        claim=citation.claim_text,
        source_content=source_excerpt,
        similarity_score=similarity_score,
    )


def validate_all(
    citations: List[Citation],
    sources: List[RetrievedSource],
) -> List[ValidationResult]:
    """Validate all citations against their sources."""
    
    print(f"\nValidating {len(citations)} citations...")

    source_map: Dict[str, RetrievedSource] = {s.reference_id: s for s in sources}
    results: List[ValidationResult] = []

    for i, citation in enumerate(citations, 1):
        print(f"  [{i}/{len(citations)}] {citation.id}...")

        source = source_map.get(citation.id)
        
        if source is None:
            result = ValidationResult(
                citation_id=citation.id,
                claim=citation.claim_text,
                status="not_found",
                confidence=1.0,
                reasoning="No matching reference found.",
                suggestion="Check citation ID matches bibliography.",
            )
        else:
            result = validate_single(citation, source)

        results.append(result)

    # Summary
    summary: Dict[str, int] = {}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1

    print("\nValidation summary:")
    for status, count in sorted(summary.items()):
        print(f"  {status}: {count}")

    return results
