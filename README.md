# Citation Validator

Automated citation verification for academic papers using multi-source retrieval and LLM-based validation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What It Does

```
PDF Paper → Extract Citations → Retrieve Sources → Validate Claims → Report
```

The system parses academic PDFs, retrieves cited sources via APIs (DBLP, arXiv, Semantic Scholar) and browser automation, then uses embeddings + LLM reasoning to verify whether claims accurately represent their sources.

**Example Output:**
```
Paper: Attention Is All You Need
Citations: 43/50 validated

┌────────────────┬───────┬────────────┐
│ Status         │ Count │ Percentage │
├────────────────┼───────┼────────────┤
│ Supported      │    11 │      25.6% │
│ Misrepresented │     1 │       2.3% │
│ Unverifiable   │    25 │      58.1% │
│ Not Found      │     6 │      14.0% │
└────────────────┴───────┴────────────┘

Desk Rejection Risk: 2/5 (Low)
```

## Quick Start

```bash
# Install
pip install -r requirements.txt
playwright install chromium

# Configure
cp .env.example .env
# Add OPENAI_API_KEY to .env

# Run
python main.py validate paper.pdf
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline                                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Parser        │ PyMuPDF extracts citations & references     │
│  2. Retriever     │ DBLP → arXiv → Semantic Scholar → Browser   │
│  3. Validator     │ FAISS embeddings + GPT-4/Claude reasoning   │
│  4. Reporter      │ JSON + CLI output with recommendations      │
└─────────────────────────────────────────────────────────────────┘
```

### Retrieval Hierarchy

| Priority | Source | Rate Limit | Coverage |
|----------|--------|------------|----------|
| 1 | DBLP | None | CS/ML papers |
| 2 | arXiv | ~3/sec | Preprints |
| 3 | Semantic Scholar | ~10/min | Broad |
| 4 | Browser (Playwright) | N/A | Fallback |

### Validation Categories

| Status | Meaning |
|--------|---------|
| **Supported** | Claim verified against source |
| **Misrepresented** | Claim contradicts source |
| **Unverifiable** | Source found but insufficient content |
| **Not Found** | Source retrieval failed |

## Usage

```bash
# Full validation
python main.py validate paper.pdf

# Preview extraction (no API calls)
python main.py preview paper.pdf

# Custom output location
python main.py validate paper.pdf --output results.json

# Limit citations checked
python main.py validate paper.pdf --max-citations 20
```

## Configuration

```bash
# .env
OPENAI_API_KEY=sk-...              # Required
ANTHROPIC_API_KEY=sk-ant-...       # Optional (for Claude)
DEFAULT_LLM_PROVIDER=openai        # openai | anthropic
SIMILARITY_THRESHOLD=0.75          # Embedding match threshold
MAX_CITATIONS_TO_CHECK=50
```

## Project Structure

```
paper-validator/
├── agents/
│   ├── parser.py          # PDF → citations + references
│   ├── retriever.py       # Multi-source paper retrieval
│   └── validator_agent.py # Embedding search + LLM validation
├── tools/
│   ├── pdf_tools.py       # PyMuPDF wrapper
│   ├── web_tools.py       # API clients (DBLP, arXiv, SS)
│   ├── browser_tools.py   # Playwright automation
│   └── embedding_tools.py # OpenAI embeddings + FAISS
├── workflows/
│   └── citation_workflow.py
├── main.py                # CLI (Typer)
├── schemas.py             # Pydantic models
└── config.py
```

## Output Format

```json
{
  "paper_title": "...",
  "citations_checked": 43,
  "results": [
    {
      "citation_id": "[13]",
      "claim": "LSTMs have been established as state of the art...",
      "status": "supported",
      "confidence": 0.95,
      "source_excerpt": "Long Short-Term Memory (LSTM) networks...",
      "reasoning": "Source confirms LSTM significance.",
      "similarity_score": 0.801
    }
  ],
  "summary": {
    "supported_count": 11,
    "misrepresented_count": 1,
    "unverifiable_count": 25,
    "not_found_count": 6,
    "desk_rejection_risk": 2
  }
}
```

## Limitations

See [Limitations.md](Limitations.md) for details.

**Key constraints:**
- Abstract-only retrieval (no full-text) → high unverifiable rate
- Semantic Scholar rate limits (429 errors)
- Cannot access paywalled content
- Table data sometimes extracted as claims

## Performance

Tested on "Attention Is All You Need" (Vaswani et al., 2017):

| Metric | Value |
|--------|-------|
| Retrieval Success | 86% (37/43) |
| Processing Time | ~4.5 min |
| Supported | 26% |
| Unverifiable | 58% |

See [RESULTS_REPORT.md](RESULTS_REPORT.md) for full analysis.

## Dependencies

- Python 3.10+
- PyMuPDF (PDF parsing)
- LangChain + OpenAI/Anthropic (LLM)
- FAISS (vector search)
- Playwright (browser automation)
- httpx (async HTTP)

## License

MIT

## Acknowledgments

Built with [LangChain](https://langchain.com/), [Semantic Scholar API](https://www.semanticscholar.org/), [DBLP](https://dblp.org/), [Playwright](https://playwright.dev/), and [FAISS](https://github.com/facebookresearch/faiss).

---

*This is a proof-of-concept for academic research. Always manually verify critical citations.*
