# Limitations & Next Steps

> **TL;DR:** 86% retrieval, 26% verified, 58% unverifiable. Main blockers: abstract-only content, table data as claims, rate limits.

---

## Critical Issues (High Impact)

| Issue | Impact | Example | Fix |
|-------|--------|---------|-----|
| **Abstract-only retrieval** | 58% unverifiable | BLEU scores not in abstracts | Integrate Unpaywall, Semantic Scholar full-text API |
| **Table data as claims** | False unverifiable | `"10²⁰ GNMT + RL [38] 39.92..."` | Add table detection, skip citations in tables |
| **Wrong paper retrieved** | False misrepresentation | [33] Dropout → AID paper | Stricter title matching (>50% overlap) |

---

## Medium Priority

| Issue | Impact | Fix |
|-------|--------|-----|
| Semantic Scholar 429 errors | 6 papers hit rate limit | Add caching layer |
| Google Scholar CAPTCHA | Browser fallback fails | Use SerpAPI or institutional proxy |
| Title truncation | [8] parsed as "Smith" | Fix reference regex parsing |
| Affiliations as authors | "Google Brain" in author list | Filter known affiliation patterns |

---

## Low Priority / Future Work

| Issue | Fix |
|-------|-----|
| No ground truth dataset | Build 100-pair golden set for F1 evaluation |
| Single paper tested | Validate on 10+ papers across domains |
| No caching | Store successful retrievals in SQLite |
| Single-hop only | Add recursive citation chain verification |

---

## Quick Wins

1. ✅ **Table detection** — Skip claims with `10^x` patterns
2. ✅ **Affiliation filter** — Blocklist "Google", "Microsoft", "University"  
3. ✅ **Caching** — SQLite for retrieved abstracts
4. ✅ **Title matching** — Increase threshold to 50%

## Requires External Access

1. ❌ **Full-text retrieval** — Needs Unpaywall API key or institutional access
2. ❌ **Google Scholar** — Needs SerpAPI ($50/mo) or proxy
3. ❌ **Paywalled papers** — Needs institutional credentials

---

## Current Performance

```
Retrieval:    37/43 (86%) ✓
Supported:    11/43 (26%)
Unverifiable: 25/43 (58%) ← Main problem
Misrepresented: 1/43 (2%) ← False positive
Not Found:     6/43 (14%)
```

**Root cause of 58% unverifiable:** Abstracts don't contain specific numbers (BLEU scores, FLOPs, hyperparameters). Need full-text access.

---

*November 2025*