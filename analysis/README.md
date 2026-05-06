# Agentic ErrorMap — Analysis Artifacts

Final analysis from running ErrorMap on 4,551 failed agent trajectories
(5 agent harnesses × 5 LM backbones × 6 benchmarks; gpt-5.5 as judge).

## Files

| File | What |
|---|---|
| `prevalence.{csv,png}` | Top-level failure category prevalence after cleanup |
| `crosstab_{model,agent,benchmark}.{csv,png}` | Category × axis cross-tabs (raw + column-normalized %) |
| `breakdown_{model,agent,benchmark}.csv` | Top-8 categories per group (paper-style) |
| `per_{model,agent,benchmark}_signatures.png` | Side-by-side bar charts per model/agent/benchmark |
| `taxonomy_mapping.csv` | Mapping of our 27 categories to ErrorAtlas-17 + AgentErrorTaxonomy-17, with novelty annotation |
| `judge_validation.csv` | 100-record judge accuracy validation (paper's protocol) |

## Reproduction scripts

| Script | What it does |
|---|---|
| `run_pipeline.py` | Full ErrorMap end-to-end (Stage 1 + Stage 2) |
| `cleanup.py` | Manual taxonomy cleanup pass (collapses Other depth-2 into top-level synonyms) |
| `per_model_signatures.py` | Generates the side-by-side signature bar charts |
| `validate_judge.py` | Runs paper's judge-accuracy validation |

## Headline numbers

- **4,551 failures** classified across 6 benchmarks
- **27 cleaned top-level categories** (51 with long-tail)
- **2.1% Other** — better than paper's 4.8%
- **85% judge agreement** (paper: 92%) — lower because our categories are more granular
- **42% of all failures fall in 14 NOVEL categories** not present in either ErrorAtlas-17 or AgentErrorTaxonomy-17

### Novel categories (your contribution)

| Category | n | % |
|---|---|---|
| Premature Termination | 232 | 8.1% |
| Search Recovery & Adaptation | 183 | 6.4% |
| Escalation Handling | 151 | 5.3% |
| Diagnostic Protocol Error | 118 | 4.1% |
| Validation Recovery Failure | 104 | 3.6% |
| Remediation Omission | 81 | 2.8% |
| Finalization Protocol Error | 73 | 2.5% |
| Unauthorized Action Execution | 69 | 2.4% |
| Query Formulation Error | 60 | 2.1% |
| Confirmation Handling Error | 43 | 1.5% |
| Authentication Handling Error | 36 | 1.3% |
| User Communication Error | 21 | 0.7% |
| Instruction Following Error | 19 | 0.7% |
| Looping Behavior | 11 | 0.4% |
