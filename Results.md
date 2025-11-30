# Citation Validation Results Report

**Test Paper:** Attention Is All You Need (Vaswani et al., 2017)  
**Validation Date:** November 30, 2025  
**Pipeline Version:** 1.0.0  
**Processing Time:** 267 seconds (~4.5 minutes)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Citations in Paper | 50 |
| Citations Validated | 43 (86%) |
| Retrieval Success Rate | 37/43 (86%) |
| Supported | 11 (25.6%) |
| Misrepresented | 1 (2.3%) |
| Unverifiable | 25 (58.1%) |
| Not Found | 6 (14.0%) |
| **Desk Rejection Risk** | **2/5 (Low)** |

---

## Validation Results Breakdown

### ✓ Supported Citations (11)

These citations were successfully retrieved and verified against source content.

| Citation | Reference | Confidence | Similarity |
|----------|-----------|------------|------------|
| [1] | Layer Normalization (Ba et al.) | 0.90 | 0.779 |
| [7] | Empirical Evaluation of Gated Recurrent Neural Networks (Chung et al.) | 0.90 | 0.829 |
| [9] | Convolutional Sequence to Sequence Learning (Gehring et al.) | 0.90 | 0.819 |
| [10] | Generating Sequences with Recurrent Neural Networks (Graves) | 0.90 | 0.734 |
| [11] | Deep Residual Learning for Image Recognition (He et al.) | 0.95 | 0.799 |
| [13] | Long Short-Term Memory (Hochreiter & Schmidhuber) | 0.95 | 0.801 |
| [18] | Neural Machine Translation in Linear Time / ByteNet (Kalchbrenner et al.) | 0.90 | 0.785 |
| [21] | Factorization Tricks for LSTM Networks (Kuchaiev & Ginsburg) | 0.90 | 0.730 |
| [30] | Using the Output Embedding to Improve Language Models (Press & Wolf) | 0.90 | 0.794 |
| [32] | Outrageously Large Neural Networks: The Sparsely-Gated MoE (Shazeer et al.) | 0.90 | 0.831 |
| [34] | End-to-End Memory Networks (Sukhbaatar et al.) | 0.90 | 0.867 |

**Example Validation:**

> **Citation [13] - LSTM**  
> **Claim:** "Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches..."  
> **Source Excerpt:** "In the 1990s, the constant error carousel and gating were introduced as the central ideas of the Long Short-Term Memory (LSTM)."  
> **Status:** ✓ Supported (Type A - Definitional)  
> **Reasoning:** The claim is definitional, mentioning the existence and establishment of LSTMs. The source confirms the introduction and significance of LSTMs.

---

### ✗ Misrepresented Citations (1)

| Citation | Reference | Issue |
|----------|-----------|-------|
| [33] | Dropout: A Simple Way to Prevent Neural Networks from Overfitting | Wrong paper retrieved |

**Analysis:**

> **Citation [33]**  
> **Expected:** Srivastava et al. (2014) - Original Dropout paper  
> **Retrieved:** Paper about "AID" (Adaptive Input Dropout) - a dropout variant  
> **Claim:** "We apply dropout [33] to the output of each sub-layer..."  
> **Source Excerpt:** "Unlike Dropout, AID generates subnetworks by applying Dropout with different probabilities..."  
> **Status:** ✗ Misrepresented (False Positive)

**Root Cause:** Title matching retrieved a related but incorrect paper. The original Dropout paper (Srivastava et al., JMLR 2014) was not in DBLP's top results, and a paper discussing dropout variants was returned instead.

**Note:** This is a **pipeline error**, not an actual citation problem in the Transformer paper. The original Dropout citation is correct.

---

### ? Unverifiable Citations (25)

Citations where source was retrieved but content was insufficient to verify the specific claim.

**Primary Causes:**

| Cause | Count | Example |
|-------|-------|---------|
| Abstract doesn't contain specific numbers | 12 | BLEU scores, FLOPs in Table 2 |
| Claim extracted from table data | 8 | Scientific notation values |
| Source discusses topic but not exact claim | 5 | Positional encodings, complexity analysis |

**Examples:**

> **Citation [3] - Byte Pair Encoding**  
> **Claim:** "Sentences were encoded using byte-pair encoding [3], which has a shared source-target vocabulary of about 37000 tokens."  
> **Source Excerpt:** "...scale analysis of NMT architecture hyperparameters..."  
> **Status:** ? Unverifiable  
> **Reason:** Source abstract discusses NMT hyperparameters but doesn't mention the specific vocabulary size (37000 tokens). This detail is likely in the paper's methodology section, not the abstract.

> **Citation [38] - GNMT BLEU Scores**  
> **Claim:** "GNMT + RL [38] 39.92 2.3 · 10¹⁹ 1.4 · 10²⁰..."  
> **Source Excerpt:** "On the WMT'14 English-to-French and English-to-German benchmarks, GNMT achieves competitive results..."  
> **Status:** ? Unverifiable  
> **Reason:** Claim contains specific BLEU scores and FLOPs from Table 2. Abstract confirms competitive results but doesn't contain exact numbers.

---

### ✗ Not Found Citations (6)

Sources that could not be retrieved through any method.

| Citation | Reference | Year | Likely Cause |
|----------|-----------|------|--------------|
| [12] | Gradient Flow in Recurrent Nets | 2001 | Book chapter, not indexed in DBLP/arXiv |
| [25] | Penn Treebank | 1993 | Dataset paper, older publication |
| [40] | Fast and Accurate Shift-Reduce Constituent Parsing | 2013 | ACL paper, rate-limited during retrieval |
| [14] | Self-Training PCFG Grammars | 2009 | EMNLP paper, browser fallback failed |
| [26] | Effective Self-Training for Parsing | 2006 | Older NLP paper, not in accessible repositories |
| [8] | Recurrent Neural Network Grammars | 2016 | Title parsing error ("Smith" extracted instead of full title) |

---

## Retrieval Performance

### Retrieval Methods Used

The console output shows successful retrievals via:

- **DBLP** (primary): `dblp_doi` and `dblp_scrape` - majority of retrievals
- **arXiv** (secondary): `arxiv_search` and `scrape_arxiv` - fallback for papers not in DBLP
- **Browser fallback**: All browser attempts failed due to Google Scholar CAPTCHA

**Total: 37/43 successful retrievals (86%)**

### API Rate Limiting

Semantic Scholar returned HTTP 429 errors after ~10 requests:

```
Client error '429' for url 'https://api.semanticscholar.org/graph/v1/paper/search?...'
```

**Impact:** 6 papers fell back to browser retrieval, which failed due to Google Scholar CAPTCHA.

### Retrieval Hierarchy Performance

```
┌─────────────────────────────────────────────────────────────┐
│                    Retrieval Funnel                         │
├─────────────────────────────────────────────────────────────┤
│  43 citations to retrieve                                   │
│    ↓                                                        │
│  23 resolved via DBLP (53%)          ← Primary source       │
│    ↓                                                        │
│  14 resolved via arXiv (33%)         ← Secondary source     │
│    ↓                                                        │
│   0 resolved via browser (0%)        ← CAPTCHA blocked      │
│    ↓                                                        │
│   6 not found (14%)                  ← Retrieval failure    │
└─────────────────────────────────────────────────────────────┘
```

---

## Claim Extraction Quality

### Issues Identified

**1. Table Data Extracted as Claims**

Several "claims" are actually table rows containing BLEU scores and FLOPs:

```
Claim: "] 1.0 · 10²⁰ GNMT + RL [38] 39.92 2.3 · 10¹⁹ 1.4 · 10²⁰ ConvS2S [9] 25.16..."
```

These are not verifiable claims—they are comparative results that would require full-text access to verify.

**2. Section Headers in Claims**

Some claims include section headers:

```
Claim: "Introduction Recurrent neural networks, long short-term memory [13]..."
```

**3. Truncated Titles**

Reference [8] parsed as "Smith" instead of "Recurrent Neural Network Grammars":

```
Retrieving: [8] - Smith...
```

---

## Validation Prompt Performance

### Citation Type Classification

| Type | Description | Count | Typical Outcome |
|------|-------------|-------|-----------------|
| A | Definitional ("LSTMs exist") | 8 | Supported |
| B | Methodological ("We use X from [Y]") | 6 | Supported |
| C | Background ("Prior work shows...") | 4 | Supported/Unverifiable |
| D | Factual ("BLEU=25.16") | 25 | Unverifiable |

**Observation:** Type D (factual) claims have the highest unverifiable rate because abstracts rarely contain specific experimental results.

---

## Recommendations

### For This Paper

1. **[33] Dropout:** False positive—pipeline error, not citation error
2. **[12], [25], [14], [26], [40], [8]:** Manually verify these 6 references (older NLP papers)
3. **Table citations:** Consider excluding table-based citations from validation

### For Pipeline Improvement

| Priority | Issue | Suggested Fix |
|----------|-------|---------------|
| High | Table data as claims | Add table detection, skip citations in tables |
| High | Semantic Scholar rate limits | Implement caching, reduce API calls |
| Medium | Google Scholar CAPTCHA | Use institutional proxy or alternative sources |
| Medium | Abstract-only retrieval | Integrate full-text sources (Unpaywall, PubMed Central) |
| Low | Author extraction | Filter affiliation patterns more aggressively |

---

## Appendix: Full Citation Status

| ID | Status | Confidence | Similarity |
|----|--------|------------|------------|
| [1] | supported | 0.90 | 0.779 |
| [2] | unverifiable | 0.50 | 0.658 |
| [3] | unverifiable | 0.70 | 0.643 |
| [6] | unverifiable | 0.70 | 0.764 |
| [7] | supported | 0.90 | 0.829 |
| [8] | not_found | - | - |
| [8] | unverifiable | 0.60 | 0.711 |
| [8] | unverifiable | 0.60 | 0.751 |
| [9] | supported | 0.90 | 0.819 |
| [9] | unverifiable | 0.50 | 0.703 |
| [9] | unverifiable | 0.70 | 0.785 |
| [9] | unverifiable | 0.70 | 0.763 |
| [9] | unverifiable | 0.60 | 0.729 |
| [10] | supported | 0.90 | 0.734 |
| [11] | supported | 0.95 | 0.799 |
| [12] | not_found | - | - |
| [13] | supported | 0.95 | 0.801 |
| [14] | not_found | - | - |
| [16] | unverifiable | 0.50 | 0.724 |
| [18] | supported | 0.90 | 0.785 |
| [18] | unverifiable | 0.70 | 0.704 |
| [20] | unverifiable | 0.60 | 0.755 |
| [21] | supported | 0.90 | 0.730 |
| [23] | unverifiable | 0.70 | 0.749 |
| [25] | not_found | - | - |
| [26] | not_found | - | - |
| [27] | unverifiable | 0.60 | 0.742 |
| [29] | unverifiable | 0.60 | 0.681 |
| [30] | supported | 0.90 | 0.794 |
| [31] | unverifiable | 0.70 | 0.702 |
| [32] | supported | 0.90 | 0.831 |
| [33] | misrepresented | 0.90 | 0.702 |
| [34] | supported | 0.90 | 0.867 |
| [36] | unverifiable | 0.70 | 0.714 |
| [37] | unverifiable | 0.70 | 0.746 |
| [38] | unverifiable | 0.60 | 0.773 |
| [38] | unverifiable | 0.70 | 0.789 |
| [38] | unverifiable | 0.70 | 0.758 |
| [39] | unverifiable | 0.70 | 0.790 |
| [39] | unverifiable | 0.70 | 0.788 |
| [40] | not_found | - | - |

**Note:** Some citation IDs appear multiple times because the same reference is cited in different contexts throughout the paper (e.g., [9] ConvS2S is cited 5 times for different claims).

---

## Conclusion

The pipeline successfully demonstrates adaptive, multi-source retrieval for citation validation:

- **86% retrieval rate** using tiered DBLP → arXiv → browser strategy
- **25.6% supported** with high confidence
- **2.3% misrepresented** (1 false positive due to retrieval error)
- **58.1% unverifiable** primarily due to abstract-only content

The high unverifiable rate reflects a known limitation: most APIs return abstracts, not full text. Factual claims (specific numbers, BLEU scores) require full paper access to verify.

**Key Finding:** The single "misrepresented" citation is a pipeline error (wrong paper retrieved), not an actual citation problem in the Transformer paper. This highlights the importance of title matching verification—a feature that was improved during development but still has edge cases.

---

*Report generated by Citation Validation Pipeline v1.0.0*