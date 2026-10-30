# Long-document company extraction using candidate-focused evidence

> Follow-up: all 100 paired outputs now have a source-reviewed semantic evaluation; see [the reviewed results](#assistant-reviewed-evaluation-of-all-100-documents). The original automatic proxy results below are preserved.

## Main findings

The evidence method completed all 100 held-out documents, including 59 above 32K estimated tokens, without context errors or abstentions. The longest was 285,617 estimated tokens. This supports the feasibility of bounded-context selection over long reports; it does not validate evidence sufficiency for every decision.

The name-match proxy was 81/100 versus 82/100 for Gemini. On the >32K subset it was 45/59 versus 46/59. These small differences are not evidence of semantic equivalence or statistical superiority, particularly before alias review.

API cost fell by 87.9% ($0.180434 vs $1.496941), with local compute priced at $0. End-to-end execution was 6.21× slower (802.5s vs 129.2s). Local extraction remains the principal bottleneck.

Evidence Jev made 192 requests and reported 4,296,041 input tokens across all batches/rounds; Gemini made 100 requests and reported 5,978,231 prompt tokens. These token counts use different provider tokenizers and should not be interpreted as an exact text-compression ratio. The cost advantage also reflects Jev's lower token price. No held-out document invoked the NONE-triggered expansion, so the expansion path has offline test coverage but no empirical quality evidence from this run.

## Scope and method

This experiment tests whether Jev can select a report’s subject company without receiving the full report. GLiNER Small scans the full original text on MPS using batch size 16, 320 word/punctuation-unit windows and overlap 64. No documents, candidates, or text windows are discarded based on length. Deterministic code collects a 4,000-character opening and up to two occurrence-centered passages per candidate (180 characters on each side). It starts with the first mention, then chooses the mention farthest from those already selected; overlapping intervals are merged. Names, occurrence counts, relative first position and original offsets accompany the verbatim passages.

Candidates are packed into choice questions by both token budget (10K estimated question tokens) and option count (254 candidates + NONE). Questions share the opening state and are packed into requests under conservative 24K state-plus-question and 48K total-token guards. Winners advance with their evidence; source offsets are copied after selection. If selection returns NONE, one expansion pass uses up to four dispersed mentions per candidate. There is no uncalibrated confidence cutoff, generated summary, GT-guided retrieval, or automatic API retry. Evidence expansion can add nothing for candidates with few mentions.

Evidence rules and source hashes were frozen before the held-out run. The pilot reuses exactly the same saved candidates for both selectors. The held-out evaluation uses the entire existing 100-document test split, disjoint from the 11 pilot documents, including annual, financial, half-year and sustainability reports. This evaluates standalone company extraction; no upstream sustainability filter is applied. Gemini 3.1 Flash-Lite receives each full document without candidates, uses minimal thinking and the same direct-extraction baseline prompt as earlier experiments. It runs after local evidence extraction finishes, to avoid overlapping benchmark workloads.

## Paired 11-document pilot

| Method | Name-match proxy | Selected | Known API cost (USD) | Calls incl. retry/failures | Sum of call latency (s) |
|---|---:|---:|---:|---:|---:|
| Full-document Jev | 8/11 | 11/11 | $0.008866410 | 12 | 4.193 |
| Evidence-based Jev | 8/11 | 11/11 | $0.007918512 | 14 | 5.844 |

The full-document pilot had a transport failure; its failed document was explicitly retried in a separate artifact directory. The table includes both attempts’ available charges and elapsed call time, and uses the retry result for matching. Unknown-cost calls are reported in session accounting. Pilot extraction was reused and is excluded from both selector measurements. One run per selector cannot establish a statistically reliable quality difference.

| ID | Existing GT | Full-document output | Evidence output |
|---|---|---|---|
| 4920 | Volvo Car Group | Volvo Car Group | Volvo Car Group |
| 8870 | LGIM | LGIM | LGIM |
| 4779 | Amega | Ammega Group | Ammega |
| 5962 | Deloite | Deloitte Audit & Assurance | Deloitte Audit & Assurance |
| 8667 | News Corp | News Corporation | News Corp |
| 3678 | Nuance | Nuance | Nuance |
| 5466 | HEINEKEN UK | HEINEKEN UK | HEINEKEN UK |
| 4546 | Nuday AS | Nuuday A/S | Nuuday |
| 7523 | Enpro | Enpro Industries, Inc. | Enpro |
| 8170 | DigitalBridge | DigitalBridge | DigitalBridge |
| 4922 | Volvo Group | Volvo Group | AB Volvo |

## Held-out 100-document results

| Method | Name-match proxy | Selected | Abstentions | Errors | Known API cost (USD) | Pipeline time (s) |
|---|---:|---:|---:|---:|---:|---:|
| GLiNER Small + evidence Jev | 81/100 | 100 | 0 | 0 | $0.180433722 | 802.545 |
| Gemini 3.1 Flash-Lite, full document | 82/100 | 100 | 0 | 0 | $1.496941250 | 129.225 |

Evidence-method candidate coverage proxy: **100/100**. Final name-match proxy conditional on candidate coverage: **81/100**. Evidence expansion occurred in 0 document results. Local extraction summed to 676.868s, selection/copy to 111.738s; model loading was 3.499s separately. The held-out evidence runner is sequential, not the earlier overlap scheduler; these timings measure this implementation rather than an optimized serving comparison.

### Results by original document length

Lengths below are cl100k_base estimates, not the provider tokenizer. A document above 32K estimated tokens illustrates the intended long-document regime; this is not a provider-confirmed full-context rejection. Even shorter documents may exceed context once questions are added.

| Estimated document tokens | Documents | Candidate coverage proxy | Evidence Jev match | Gemini match | Evidence API cost (USD) |
|---|---:|---:|---:|---:|---:|
| ≤24K | 30 | 30/30 | 28/30 | 26/30 | $0.019728492 |
| 24K–32K | 11 | 11/11 | 8/11 | 10/11 | $0.011875584 |
| 32K–64K | 35 | 35/35 | 26/35 | 27/35 | $0.052793118 |
| 64K–128K | 14 | 14/14 | 11/14 | 11/14 | $0.039536826 |
| >128K | 10 | 10/10 | 8/10 | 8/10 | $0.056499702 |

Maximum observed evidence request sizes: **11,742** estimated tokens for state + longest question and **47,685** for state + all questions. Every saved evidence passage was checked against its original source offsets, and every choice question had ≤255 options. The longest test document had **285,617** estimated tokens.

## Interpretation and limitations

- Name-match scores use the existing MOAR similarity function at threshold ≥0.5, aligned by document ID. They are automatic proxies; the follow-up source review below separately adjudicates company identity. Errors and abstentions count as unmatched.
- Candidate coverage uses that same approximate matching rule. It is not strict entity recall. Evidence sufficiency has **not** been independently annotated: finding a candidate name in a passage does not prove that the passage establishes report subjecthood.
- Eliminating full-document context overflow does not establish unchanged semantic accuracy. Selection can fail because GLiNER misses the name, deterministic retrieval omits the decisive passage, or Jev chooses incorrectly. These causes require source/evidence review.
- Full-document GLiNER scanning still scales with document length; evidence reduction primarily bounds Jev input, not local extraction time. More candidates can require additional requests and tournament rounds.
- One fixed test split and one run per method are not a generalization or statistical significance guarantee. Batch/round boundaries and API response variability may affect outcomes.
- Monetary comparisons count local computation as $0 and use published paid-tier token prices (Jev $0.042/M input; Gemini $0.25/M input, $0.025/M cached input, $1.50/M output including thinking). These are token-equivalent costs, not verified invoices. Unknown usage is not assumed free.

## Session accounting and artifacts

Total known API cost for this evidence experiment, including pilot, explicit pilot retry and both test methods: **$1.694159894**, plus any unreported charges from **1** calls without usage. Previous experiments are excluded. Local processing is $0 by convention.

[Frozen policy and source hashes](runs/long_document_evidence_v1/config.json) · [Per-document evaluation](runs/long_document_evidence_v1/evaluation.json) · [Evidence review packets (unannotated)](runs/long_document_evidence_v1/evidence_review.json) · [Evidence outputs](runs/long_document_evidence_v1/test_evidence/results.json) · [Jev request/response ledger](runs/long_document_evidence_v1/test_evidence/calls.jsonl) · [Gemini outputs](runs/long_document_evidence_v1/test_gemini/results.json) · [Gemini ledger](runs/long_document_evidence_v1/test_gemini/calls.jsonl)

## Per-document held-out outputs

| ID | Est. tokens | Candidates | Existing GT | Evidence output | Evidence match | Gemini output | Gemini match |
|---|---:|---:|---|---|---|---|---|
| 9455 | 54,865 | 157 | Munich Re | Munich Re | True | Munich Re | True |
| 5281 | 17,295 | 50 | NSI NV | NSI | True | NSI NV | True |
| 6815 | 19,911 | 56 | Hiperbaric | Hiperbaric | True | Hiperbaric | True |
| 4692 | 185,595 | 1007 | ThyssenKrup AG | ThyssenKrupp AG | True | ThyssenKrupp AG | True |
| 3523 | 18,356 | 90 | Thrive Home Builders | Thrive Home Builders | True | Thrive Home Builders | True |
| 4209 | 18,070 | 80 | Life Healthcare | Life Healthcare | True | Life Healthcare | True |
| 7967 | 84,739 | 106 | Swis Re | Swiss Re | True | Swiss Re | True |
| 8842 | 16,774 | 70 | Domino's | Domino’s | True | Domino’s | True |
| 6948 | 25,267 | 94 | Meser | Messer Group | False | Messer | True |
| 6151 | 29,574 | 130 | Credit Suisse | Credit Suisse | True | Credit Suisse | True |
| 4493 | 15,741 | 52 | Trex | Trex Company, Inc. | False | Trex Company, Inc. | False |
| 2610 | 54,798 | 202 | Chevron | Chevron | True | Chevron | True |
| 2981 | 73,554 | 157 | Zignago Vetro | Zignago Vetro | True | Zignago Vetro | True |
| 5539 | 28,399 | 74 | Vonovia | Vonovia SE | True | Vonovia | True |
| 3323 | 52,184 | 196 | National Bank of Canada | National Bank of Canada | True | National Bank of Canada | True |
| 6396 | 13,332 | 38 | Convenience Retail Asia Limited | Convenience Retail Asia Limited | True | Convenience Retail Asia Limited | True |
| 6225 | 36,120 | 76 | Osaka Organic Chemical Industry Ltd. | Osaka Organic Chemical Industry Ltd. | True | Osaka Organic Chemical Industry Ltd. | True |
| 6740 | 84,687 | 230 | Givaudan | Givaudan | True | Givaudan | True |
| 5636 | 44,678 | 106 | Melrose Industries PLC | Melrose Industries PLC | True | Melrose Industries PLC | True |
| 8583 | 36,071 | 159 | ENGIE | ENGIE | True | ENGIE | True |
| 6672 | 22,251 | 41 | Electrolux Profesional | Electrolux Professional | True | Electrolux Professional | True |
| 5388 | 113,979 | 218 | Enel | Enel Russia PJSC | False | Enel Russia PJSC | False |
| 3820 | 285,617 | 515 | Lucky Core Industries Limited | Lucky Core Industries Limited | True | Lucky Core Industries Limited | True |
| 4720 | 20,251 | 175 | MN | MN | True | MN | True |
| 1070 | 98,759 | 264 | Northvolt | Northvolt | True | Northvolt | True |
| 3773 | 15,414 | 71 | Chart Industries Inc. | Chart Industries, Inc. | True | Chart Industries, Inc. | True |
| 3251 | 25,952 | 169 | Telekom Srbija Group | Telekom Srbija Group | True | Telekom Srbija Group | True |
| 7675 | 10,884 | 36 | AGCO | AGCO | True | AGCO | True |
| 5733 | 27,187 | 83 | GIP ATLAS Holdings Ltd | ATLAS Infrastructure | False | ATLAS Infrastructure | False |
| 9017 | 18,195 | 81 | Bionics Institute | Bionics Institute | True | Bionics Institute | True |
| 4003 | 24,049 | 86 | Bolton Group | BOLTON GROUP | True | Bolton Group | True |
| 4167 | 39,835 | 144 | TELUS | TELUS Corporation | True | TELUS Corporation | True |
| 3432 | 19,903 | 73 | Tata Chemicals Limited | Tata Chemicals Limited | True | Tata Chemicals Limited | True |
| 9969 | 54,135 | 208 | Ferero Group | Ferrero Group | True | Ferrero Group | True |
| 8370 | 39,556 | 120 | UPL | UPL Limited | True | UPL Limited | True |
| 7846 | 47,514 | 156 | Siemens | Siemens | True | Siemens | True |
| 8536 | 16,620 | 125 | TPG | TPG | True | TPG | True |
| 6226 | 68,162 | 215 | SCG Chemicals Public Company Limited | SCG Chemicals Public Company Limited | True | SCG Chemicals Public Company Limited | True |
| 9101 | 33,939 | 150 | Nestlé | Nestlé | True | Nestlé | True |
| 3806 | 58,572 | 180 | Alba | Alba | True | Aluminium Bahrain (Alba) | False |
| 9599 | 41,280 | 146 | Julius Baer Group | JULIUS BAER GROUP | True | JULIUS BAER GROUP | True |
| 8830 | 85,002 | 329 | PB Group Berhad | PPB Group Berhad | True | PPB Group Berhad | True |
| 5626 | 46,826 | 236 | Rufer | Ruffer LLP | False | Ruffer LLP | False |
| 9330 | 18,469 | 48 | Tele2 | Tele2 | True | Tele2 | True |
| 5966 | 49,181 | 147 | Barila | Barilla Group | False | Barilla | True |
| 8564 | 30,112 | 121 | Williams-Sonoma | Williams-Sonoma, Inc. | True | Williams-Sonoma, Inc. | True |
| 9660 | 266,581 | 674 | Stelantis NV | Stellantis N.V. | False | Stellantis N.V. | False |
| 6932 | 40,934 | 161 | Legal & General Group Plc | Legal & General Group Plc | True | Legal & General Group Plc | True |
| 6863 | 12,132 | 45 | Intesa Sanpaolo | Intesa Sanpaolo | True | Intesa Sanpaolo | True |
| 3896 | 50,744 | 174 | Grape King Bio | Grape King Bio | True | Grape King Bio | True |
| 3700 | 13,095 | 28 | Grape King Bio | Grape King Bio | True | Grape King Bio | True |
| 9277 | 47,508 | 214 | PowerGrid | Power Grid Corporation of India Limited | False | Power Grid Corporation of India Limited | False |
| 2050 | 142,468 | 338 | Epson | Seiko Epson Corporation | False | Seiko Epson Corporation | False |
| 4021 | 28,607 | 45 | Neapco | Neapco | True | Neapco | True |
| 5781 | 20,346 | 95 | Howden Group | Howden Group Holdings | True | Howden Group Holdings | True |
| 7526 | 16,606 | 37 | FinDreams Battery | FinDreams Battery | True | FinDreams Battery Company Limited | True |
| 3318 | 34,023 | 82 | Li & Fung Limited | Li & Fung Limited | True | Li & Fung Limited | True |
| 7956 | 47,265 | 125 | Veolia Environnement | VEOLIA ENVIRONNEMENT | True | VEOLIA ENVIRONNEMENT | True |
| 7103 | 42,309 | 172 | GBL | Groupe Bruxelles Lambert SA / Groep Brussel Lambert NV | False | Groupe Bruxelles Lambert | False |
| 7109 | 106,657 | 671 | T. Rowe Price | T. Rowe Price | True | T. Rowe Price | True |
| 3487 | 80,258 | 213 | PotlatchDeltic | PotlatchDeltic | True | PotlatchDeltic | True |
| 6517 | 78,817 | 233 | Pan American Silver | Pan American Silver | True | Pan American Silver | True |
| 3934 | 48,699 | 102 | Celestica | Celestica | True | Celestica | True |
| 4496 | 93,248 | 302 | Dow | Dow Inc. | False | Dow Inc. | False |
| 2929 | 52,741 | 190 | bioMerieux | bioMérieux | True | bioMérieux | True |
| 7329 | 113,573 | 316 | Sri Trang Group | Sri Trang Agro-Industry Public Company Limited | False | Sri Trang Agro-Industry Public Company Limited | False |
| 3003 | 48,531 | 116 | JD Irving, Limited | J.D. Irving, Limited | False | J.D. Irving, Limited | False |
| 7505 | 44,397 | 106 | U.S. Steel | U. S. Steel | False | United States Steel Corporation | False |
| 3289 | 46,561 | 215 | Hilton | Hilton Worldwide Holdings Inc. | False | Hilton | True |
| 9114 | 37,269 | 135 | Arconic | Arconic | True | Arconic Corporation | True |
| 7014 | 57,338 | 190 | Lamb Weston / Meijer | Lamb Weston / Meijer | True | Lamb Weston / Meijer | True |
| 7919 | 16,611 | 92 | Cardinal Health | Cardinal Health | True | Cardinal Health | True |
| 6539 | 90,878 | 287 | Beach Energy Limited | Beach Energy Limited | True | Beach Energy Limited | True |
| 4645 | 191,064 | 621 | Tesco PLC | Tesco PLC | True | Tesco PLC | True |
| 653 | 136,465 | 308 | Mitsubishi Electric | Mitsubishi Electric Group | True | MITSUBISHI ELECTRIC | True |
| 5980 | 28,157 | 167 | Diageo | Diageo | True | Diageo | True |
| 5963 | 39,770 | 160 | Conexia | Connexia Società Benefit S.r.l. | False | Connexia | True |
| 4595 | 79,122 | 211 | NT DATA Corporation | NTT DATA Corporation | True | NTT DATA Corporation | True |
| 3290 | 207,823 | 624 | Acea Group | ACEA GROUP | True | ACEA GROUP | True |
| 6908 | 14,588 | 42 | LafargeHolcim | LafargeHolcim | True | LafargeHolcim | True |
| 8756 | 12,482 | 79 | JPMorgan Chase | JPMorgan Chase & Co. | True | JPMorgan Chase & Co. | True |
| 7315 | 130,923 | 333 | Sartorius Group | Sartorius Group | True | Sartorius AG | True |
| 2667 | 49,456 | 149 | Prada Group | Prada Group | True | Prada S.p.A. | False |
| 5819 | 17,525 | 20 | Hitachi Rail | Hitachi Rail | True | Hitachi Rail | True |
| 8959 | 18,129 | 86 | Arch Capital Group Ltd. | Arch Capital Group Ltd. | True | Arch Capital Group Ltd. | True |
| 7696 | 31,219 | 95 | Dominos | Domino's Pizza, Inc. | False | Domino’s | True |
| 4986 | 35,674 | 82 | Diebold Nixdorf | Diebold Nixdorf | True | Diebold Nixdorf | True |
| 8892 | 42,327 | 133 | Experian | Experian | True | Experian | True |
| 4991 | 53,094 | 190 | PVH Corp | PVH CORP. | False | PVH Corp. | False |
| 9792 | 26,026 | 126 | Renault Group | Renault Group | True | Renault Group | True |
| 8491 | 15,238 | 101 | Shenzhen Mindray Bio medical Electronics Co., Ltd. | Shenzhen Mindray Bio-medical Electronics Co., Ltd. | True | Shenzhen Mindray Bio-medical Electronics Co., Ltd. | True |
| 8986 | 186,498 | 304 | Fermenta Biotech Limited | Fermenta Biotech Limited | True | Fermenta Biotech Limited | True |
| 3170 | 38,320 | 135 | Suma Equity | Summa Equity | True | Summa Equity | True |
| 3876 | 15,387 | 38 | TJC | TJC | True | TJC | True |
| 6832 | 193,959 | 495 | Imperial Brands PLC | Imperial Brands PLC | True | Imperial Brands PLC | True |
| 5577 | 19,166 | 79 | Sabelt | Sabelt | True | Sabelt S.p.A. | False |
| 9019 | 16,046 | 40 | Sunova | Sunnova Energy International Inc. | False | Sunnova Energy International Inc. | False |
| 5135 | 13,995 | 98 | MOMENI | MOMENI Group | True | MOMENI | True |
| 9515 | 44,585 | 192 | Altice France | Altice France group | True | Altice France | True |
| 5605 | 12,921 | 33 | Seaway7 | Seaway7 | True | Seaway 7 ASA | False |

## Assistant-reviewed evaluation of all 100 documents

Follow-up review date: 2026-09-28. **All 100 documents and both outputs per document were reviewed against source excerpts**, including targeted searches for ambiguous names. The earlier 81/100 and 82/100 results are automatic string-similarity scores, not adjudicated semantic accuracy. Original predictions and GT are unchanged.

**Rubric:** accept the report’s company, a source-supported brand/legal name, or an explicitly established reporting parent/group. Do not automatically equate arbitrary parents and subsidiaries. This scores entity identity, not completeness of the legal name or exact consolidation scope. Source-copy fidelity is measured separately.

| Metric | GLiNER Small + evidence Jev | Gemini 3.1 Flash-Lite |
|---|---:|---:|
| Confirmed correct company identity / all documents | 99/100 | 99/100 |
| Ambiguous subject-versus-publisher cases | 1 | 1 |
| Confirmed wrong-company outputs | 0 | 0 |
| Correct among 99 unambiguous documents | 99/99 | 99/99 |
| Exact contiguous source-string occurrence | 100/100 | 99/100 |
| Source occurrence after whitespace normalization | 100/100 | 100/100 |
| Automatic proxy negatives accepted after source review | 19 | 18 |

On the **59 documents above 32K estimated tokens**, both methods have 59/59 confirmed correct identities in this review. The ambiguous document is below 32K.

**ID 6151 — Credit Suisse:** this is a thematic report on plastic pollution, produced by the Credit Suisse Center for Sustainability. Both methods return Credit Suisse, matching the publisher and existing GT. Under a strict subject-company task, NONE is defensible. We therefore retain one ambiguous case for each method: 99% confirmed correct over all 100, or 100% among the 99 unambiguous cases. If the task explicitly includes publisher extraction, this case can be accepted; that scope choice should be made before the next evaluation.

**ID 7526 — FinDreams:** Gemini returns “FinDreams Battery Company Limited”; the source contains that full name across a line break. It is semantically supported and matches after whitespace normalization, but is not an exact contiguous copy. Jev’s returned “FinDreams Battery” occurs verbatim. String occurrence alone does not prove correct entity selection.

Examples of automatic false negatives include Trex Company, Inc. versus Trex; Power Grid Corporation of India Limited versus PowerGrid; and Hilton Worldwide Holdings Inc. versus Hilton. The source also resolves scope-specific labels: Enel Russia PJSC is the subject of the Enel Russia report, while ATLAS Infrastructure is the reported organization whose governing body is GIP ATLAS Holdings Ltd.

**Interpretation:** this review finds no confirmed identity-quality gap between the two methods on these 100 outputs. It strengthens the feasibility result for long-document extraction, but does not establish statistical equivalence or generalization. The original cost/latency result remains $0.180434 and 802.5s for the evidence method versus $1.496941 and 129.2s for Gemini. No new model calls were made for this review.

**Limitations:** these are post-hoc, unblinded assistant judgments from source excerpts and targeted searches, not independent human-validated gold labels or exhaustive full-document readings. The review does not separately establish that every reduced evidence package was sufficient for Jev’s decision. A blinded human review and a frozen alias/scope rubric are appropriate before presenting these numbers as benchmark accuracy.

[All 100 paired judgments, rationales, source hashes and exact source offsets](runs/long_document_evidence_v1/semantic_review.json). Offsets are zero-based, end-exclusive Python character indices into each `test_input.json` text. Regenerate this section with `python report_semantic_review.py` after editing adjudications.

### Individual source-reviewed judgments

| ID | Evidence Jev output | Gemini output | Judgment for both | Source-review rationale |
|---|---|---|---|---|
| 9455 | Munich Re | Munich Re | correct | The source opening and name-bearing excerpts identify Munich Re as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 5281 | NSI | NSI NV | correct | The source opening and name-bearing excerpts identify NSI as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 6815 | Hiperbaric | Hiperbaric | correct | The source opening and name-bearing excerpts identify Hiperbaric as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 4692 | ThyssenKrupp AG | ThyssenKrupp AG | correct | The source opening and name-bearing excerpts identify ThyssenKrupp AG as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 3523 | Thrive Home Builders | Thrive Home Builders | correct | The source opening and name-bearing excerpts identify Thrive Home Builders as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 4209 | Life Healthcare | Life Healthcare | correct | The source opening and name-bearing excerpts identify Life Healthcare as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 7967 | Swiss Re | Swiss Re | correct | The source opening and name-bearing excerpts identify Swiss Re as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 8842 | Domino’s | Domino’s | correct | The source opening and name-bearing excerpts identify Domino’s as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 6948 | Messer Group | Messer | correct | The report identifies Messer and presents Messer Group and its Messer Industries investment. Accept the reporting-company/group name; this does not mean every separately reported entity is interchangeable. |
| 6151 | Credit Suisse | Credit Suisse | ambiguous | The foreword identifies Credit Suisse Center for Sustainability as the producer of a report about plastic pollution. Both outputs match the publisher and existing GT. Neither establishes that Credit Suisse is the subject company. Mark ambiguous under the subject-company task; NONE is defensible under a strict subject-only reading. |
| 4493 | Trex Company, Inc. | Trex Company, Inc. | correct | The About Trex section explicitly identifies Trex Company, Inc.; the legal name is a valid expansion of the GT brand. |
| 2610 | Chevron | Chevron | correct | The source opening and name-bearing excerpts identify Chevron as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 2981 | Zignago Vetro | Zignago Vetro | correct | The source opening and name-bearing excerpts identify Zignago Vetro as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 5539 | Vonovia SE | Vonovia | correct | The scope explicitly covers Vonovia SE and excludes Deutsche Wohnen SE; both the short and legal name identify the reporting company. |
| 3323 | National Bank of Canada | National Bank of Canada | correct | The source opening and name-bearing excerpts identify National Bank of Canada as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 6396 | Convenience Retail Asia Limited | Convenience Retail Asia Limited | correct | The source opening and name-bearing excerpts identify Convenience Retail Asia Limited as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 6225 | Osaka Organic Chemical Industry Ltd. | Osaka Organic Chemical Industry Ltd. | correct | The source opening and name-bearing excerpts identify Osaka Organic Chemical Industry Ltd. as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 6740 | Givaudan | Givaudan | correct | The source opening and name-bearing excerpts identify Givaudan as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 5636 | Melrose Industries PLC | Melrose Industries PLC | correct | The source opening and name-bearing excerpts identify Melrose Industries PLC as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 8583 | ENGIE | ENGIE | correct | The source opening and name-bearing excerpts identify ENGIE as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 6672 | Electrolux Professional | Electrolux Professional | correct | The source opening and name-bearing excerpts identify Electrolux Professional as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 5388 | Enel Russia PJSC | Enel Russia PJSC | correct | The source is the Enel Russia report, explicitly naming Enel Russia PJSC. The returned subsidiary is more specific than the existing Enel label and is the document subject. |
| 3820 | Lucky Core Industries Limited | Lucky Core Industries Limited | correct | The source opening and name-bearing excerpts identify Lucky Core Industries Limited as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 4720 | MN | MN | correct | The source opening and name-bearing excerpts identify MN as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 1070 | Northvolt | Northvolt | correct | The source opening and name-bearing excerpts identify Northvolt as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 3773 | Chart Industries, Inc. | Chart Industries, Inc. | correct | The source opening and name-bearing excerpts identify Chart Industries, Inc. as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 3251 | Telekom Srbija Group | Telekom Srbija Group | correct | The source opening and name-bearing excerpts identify Telekom Srbija Group as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 7675 | AGCO | AGCO | correct | The source opening and name-bearing excerpts identify AGCO as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 5733 | ATLAS Infrastructure | ATLAS Infrastructure | correct | The letter addresses ATLAS Infrastructure stakeholders and calls this its stewardship report. GIP ATLAS Holdings Ltd is identified as its governing body. Accept ATLAS Infrastructure as the reported organization; do not assume these labels are universally interchangeable legal entities. |
| 9017 | Bionics Institute | Bionics Institute | correct | The source opening and name-bearing excerpts identify Bionics Institute as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 4003 | BOLTON GROUP | Bolton Group | correct | The source opening and name-bearing excerpts identify BOLTON GROUP as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 4167 | TELUS Corporation | TELUS Corporation | correct | The source opening and name-bearing excerpts identify TELUS Corporation as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 3432 | Tata Chemicals Limited | Tata Chemicals Limited | correct | The source opening and name-bearing excerpts identify Tata Chemicals Limited as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 9969 | Ferrero Group | Ferrero Group | correct | The source opening and name-bearing excerpts identify Ferrero Group as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 8370 | UPL Limited | UPL Limited | correct | The source opening and name-bearing excerpts identify UPL Limited as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 7846 | Siemens | Siemens | correct | The source opening and name-bearing excerpts identify Siemens as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 8536 | TPG | TPG | correct | The source opening and name-bearing excerpts identify TPG as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 6226 | SCG Chemicals Public Company Limited | SCG Chemicals Public Company Limited | correct | The source opening and name-bearing excerpts identify SCG Chemicals Public Company Limited as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 9101 | Nestlé | Nestlé | correct | The source opening and name-bearing excerpts identify Nestlé as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 3806 | Alba | Aluminium Bahrain (Alba) | correct | The source uses Aluminium Bahrain (Alba), supporting the abbreviation and expanded output. |
| 9599 | JULIUS BAER GROUP | JULIUS BAER GROUP | correct | The source opening and name-bearing excerpts identify JULIUS BAER GROUP as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 8830 | PPB Group Berhad | PPB Group Berhad | correct | The source opening and name-bearing excerpts identify PPB Group Berhad as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 5626 | Ruffer LLP | Ruffer LLP | correct | The source explicitly identifies Ruffer LLP; the existing GT spelling Rufer is deficient. |
| 9330 | Tele2 | Tele2 | correct | The source opening and name-bearing excerpts identify Tele2 as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 5966 | Barilla Group | Barilla | correct | The source opening and name-bearing excerpts identify Barilla Group as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 8564 | Williams-Sonoma, Inc. | Williams-Sonoma, Inc. | correct | The source opening and name-bearing excerpts identify Williams-Sonoma, Inc. as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 9660 | Stellantis N.V. | Stellantis N.V. | correct | The annual-report title names Stellantis N.V. (formerly Fiat Chrysler Automobiles N.V.); the GT spelling is deficient. |
| 6932 | Legal & General Group Plc | Legal & General Group Plc | correct | The source opening and name-bearing excerpts identify Legal & General Group Plc as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 6863 | Intesa Sanpaolo | Intesa Sanpaolo | correct | The source opening and name-bearing excerpts identify Intesa Sanpaolo as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 3896 | Grape King Bio | Grape King Bio | correct | The source opening and name-bearing excerpts identify Grape King Bio as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 3700 | Grape King Bio | Grape King Bio | correct | The source opening and name-bearing excerpts identify Grape King Bio as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 9277 | Power Grid Corporation of India Limited | Power Grid Corporation of India Limited | correct | The report explicitly spells out Power Grid Corporation of India Limited (POWERGRID). |
| 2050 | Seiko Epson Corporation | Seiko Epson Corporation | correct | Reporting scope names Seiko Epson Corporation and its group companies; Epson is defined as the group. Accept the named reporting parent. |
| 4021 | Neapco | Neapco | correct | The source opening and name-bearing excerpts identify Neapco as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 5781 | Howden Group Holdings | Howden Group Holdings | correct | The source opening and name-bearing excerpts identify Howden Group Holdings as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 7526 | FinDreams Battery | FinDreams Battery Company Limited | correct | About this Report explicitly names FinDreams Battery Company Limited, with a line break between FinDreams and Battery. Both identify the subject; Gemini normalizes whitespace, so it is not a contiguous exact string copy. |
| 3318 | Li & Fung Limited | Li & Fung Limited | correct | The source opening and name-bearing excerpts identify Li & Fung Limited as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 7956 | VEOLIA ENVIRONNEMENT | VEOLIA ENVIRONNEMENT | correct | The source opening and name-bearing excerpts identify VEOLIA ENVIRONNEMENT as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 7103 | Groupe Bruxelles Lambert SA / Groep Brussel Lambert NV | Groupe Bruxelles Lambert | correct | The report uses GBL and the statutory section names Groupe Bruxelles Lambert SA / Groep Brussel Lambert NV. Both outputs identify that reporting company. |
| 7109 | T. Rowe Price | T. Rowe Price | correct | The source opening and name-bearing excerpts identify T. Rowe Price as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 3487 | PotlatchDeltic | PotlatchDeltic | correct | The source opening and name-bearing excerpts identify PotlatchDeltic as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 6517 | Pan American Silver | Pan American Silver | correct | The source opening and name-bearing excerpts identify Pan American Silver as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 3934 | Celestica | Celestica | correct | The source opening and name-bearing excerpts identify Celestica as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 4496 | Dow Inc. | Dow Inc. | correct | The report explicitly defines Dow as Dow Inc. and its consolidated subsidiaries. |
| 2929 | bioMérieux | bioMérieux | correct | The source opening and name-bearing excerpts identify bioMérieux as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 7329 | Sri Trang Agro-Industry Public Company Limited | Sri Trang Agro-Industry Public Company Limited | correct | The report title names Sri Trang Agro-Industry Public Company Limited; accept the reporting parent of Sri Trang Group. |
| 3003 | J.D. Irving, Limited | J.D. Irving, Limited | correct | The source opening and name-bearing excerpts identify J.D. Irving, Limited as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 7505 | U. S. Steel | United States Steel Corporation | correct | The GRI organizational-profile entry spells out United States Steel Corporation, supporting both it and U. S. Steel. |
| 3289 | Hilton Worldwide Holdings Inc. | Hilton | correct | GRI 102-1 explicitly gives Hilton Worldwide Holdings Inc. (Hilton), supporting both outputs. |
| 9114 | Arconic | Arconic Corporation | correct | The report is about Arconic and identifies Arconic Corporation in its reporting/legal text. |
| 7014 | Lamb Weston / Meijer | Lamb Weston / Meijer | correct | The source opening and name-bearing excerpts identify Lamb Weston / Meijer as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 7919 | Cardinal Health | Cardinal Health | correct | The source opening and name-bearing excerpts identify Cardinal Health as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 6539 | Beach Energy Limited | Beach Energy Limited | correct | The source opening and name-bearing excerpts identify Beach Energy Limited as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 4645 | Tesco PLC | Tesco PLC | correct | The source opening and name-bearing excerpts identify Tesco PLC as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 653 | Mitsubishi Electric Group | MITSUBISHI ELECTRIC | correct | The source opening and name-bearing excerpts identify Mitsubishi Electric Group as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 5980 | Diageo | Diageo | correct | The source opening and name-bearing excerpts identify Diageo as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 5963 | Connexia Società Benefit S.r.l. | Connexia | correct | The Italian reporting/GRI statement names Connexia Società Benefit S.r.l.; the short brand and legal name refer to the reporting company. |
| 4595 | NTT DATA Corporation | NTT DATA Corporation | correct | Organizations Covered identifies NTT DATA Corporation and its group companies; the GT omits a T. |
| 3290 | ACEA GROUP | ACEA GROUP | correct | The source opening and name-bearing excerpts identify ACEA GROUP as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 6908 | LafargeHolcim | LafargeHolcim | correct | The source opening and name-bearing excerpts identify LafargeHolcim as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 8756 | JPMorgan Chase & Co. | JPMorgan Chase & Co. | correct | The report explicitly defines JPMorgan Chase & Co. as JPMorgan Chase/the Firm. |
| 7315 | Sartorius Group | Sartorius AG | correct | The notes explicitly identify Sartorius AG as ultimate parent of Sartorius Group. Accept the reporting parent or group under the entity-identity rubric, without asserting identical consolidation scope. |
| 2667 | Prada Group | Prada S.p.A. | correct | The reporting-boundary section explicitly defines Prada S.p.A. together with its subsidiaries as Prada Group. Both are accepted reporting-entity names under this rubric. |
| 5819 | Hitachi Rail | Hitachi Rail | correct | The source opening and name-bearing excerpts identify Hitachi Rail as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 8959 | Arch Capital Group Ltd. | Arch Capital Group Ltd. | correct | The source opening and name-bearing excerpts identify Arch Capital Group Ltd. as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 7696 | Domino's Pizza, Inc. | Domino’s | correct | The GRI organizational-profile entry identifies Domino’s Pizza, Inc.; the brand and legal name refer to the reported organization. |
| 4986 | Diebold Nixdorf | Diebold Nixdorf | correct | The source opening and name-bearing excerpts identify Diebold Nixdorf as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 8892 | Experian | Experian | correct | The source opening and name-bearing excerpts identify Experian as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 4991 | PVH CORP. | PVH Corp. | correct | The source opening and name-bearing excerpts identify PVH CORP. as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 9792 | Renault Group | Renault Group | correct | The source opening and name-bearing excerpts identify Renault Group as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 8491 | Shenzhen Mindray Bio-medical Electronics Co., Ltd. | Shenzhen Mindray Bio-medical Electronics Co., Ltd. | correct | The source opening and name-bearing excerpts identify Shenzhen Mindray Bio-medical Electronics Co., Ltd. as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 8986 | Fermenta Biotech Limited | Fermenta Biotech Limited | correct | The source opening and name-bearing excerpts identify Fermenta Biotech Limited as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 3170 | Summa Equity | Summa Equity | correct | The source opening and name-bearing excerpts identify Summa Equity as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 3876 | TJC | TJC | correct | The source opening and name-bearing excerpts identify TJC as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 6832 | Imperial Brands PLC | Imperial Brands PLC | correct | The source opening and name-bearing excerpts identify Imperial Brands PLC as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 5577 | Sabelt | Sabelt S.p.A. | correct | The scope explicitly says the report concerns Sabelt S.p.A. alone. |
| 9019 | Sunnova Energy International Inc. | Sunnova Energy International Inc. | correct | About Sunnova identifies Sunnova Energy International Inc.; the GT spelling Sunova is deficient. |
| 5135 | MOMENI Group | MOMENI | correct | The source opening and name-bearing excerpts identify MOMENI Group as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 9515 | Altice France group | Altice France | correct | The source opening and name-bearing excerpts identify Altice France group as the reporting organization; both returned names identify that organization. Spelling, typography, or legal-suffix differences from the existing label do not change this identity judgment. |
| 5605 | Seaway7 | Seaway 7 ASA | correct | Report footers explicitly name Seaway 7 ASA Sustainability Report 2022, supporting both the brand and legal name. |
