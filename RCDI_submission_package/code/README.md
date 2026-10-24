# RCDI: risk-calibrated, cost-aware decision intelligence with LLMs

Code for the paper "Beyond Prediction: Risk-Calibrated AI Decision Intelligence for Reliable and Cost-Aware Data-Driven Decision Making".
Three Qwen2.5 tiers (1.5B, 7B, 32B-AWQ) answer GSM8K, HotpotQA and Banking77 queries. For every query and tier the study records
whether the answer is correct and three risk signals (token log-probability, self-consistency over 5 samples, verbalised confidence).
A cascade S -> M -> L -> human is then evaluated under asymmetric costs.

Files
- `rcdi.py`      workloads, LLM client with cache, calibrators, expected-cost rule, tuned thresholds, conformal risk control
- `collect.py`   runs the models and writes `results/records.jsonl` (the only step that needs a GPU)
- `rescore.py`   re-scores GSM8K answers from the cached model outputs with the repaired extractor (no GPU, no new model calls)
- `analyze.py`   all tables, figures and statistics from the records (CPU only, a few minutes)
- `runpod.sh`    one-command GPU run: `bash runpod.sh smoke` (check) then `bash runpod.sh full`

Rescoring step (disclosed in the paper, Section 4.1 and Table S3)
The first full run scored GSM8K answers with a strict extractor that missed answers written as `\boxed{}`; this lowered GSM8K accuracy for all tiers.
After inspecting that defect, the extractor was repaired and the stored outputs were re-scored (`python rescore.py --res results --out results_rescored --gsm8k-jsonl gsm8k_test.jsonl`).
This was a post-hoc change made after seeing results. Both versions are reported: `results/` (strict) and `results_rescored/` (repaired, used in the main text).
Two further changes were made after the smoke run and before the full run: Banking77 token limit 48 -> 96 and a stricter one-line answer format, because 32B answers were truncated.
GSM8K test data comes from github.com/openai/grade-school-math (grade_school_math/data/test.jsonl) if HuggingFace is not reachable.

Reproduce the analysis without a GPU: `python analyze.py --res results --reps 200` (needs `results/records.jsonl`, `tasks.jsonl`).
Code test with synthetic data (never reported): `python collect.py --n 40 --out results_mock --mock && python analyze.py --res results_mock --reps 10`.

Settings fixed before the real run
- Primary costs (units of one mean L-answer cost): uncaught error 25, human review 5; human reviewers are assumed correct.
- Risk target alpha = 0.05 (uncaught errors per query). Splits: 40% fit, 30% calibration, 30% test, 200 random replicates.
- Cost-sensitivity grid: error cost {5, 25, 100, 400} x human cost {1, 5, 20}.
- Knowledge rules: R1 raises the error cost of predicted high-stakes Banking77 intents (x5); R2 never accepts an unparsable or truncated answer.
- Hypotheses: H1 calibrators lower ECE; H2 calibration lowers decision cost of an expected-cost rule but not versus decision-tuned thresholds;
  H3 conformal risk control keeps uncaught errors at or below alpha on average, and the correction matters only for small calibration sets;
  H4 guarantees weaken when calibration comes from other workloads; H5 the multi-signal model pays for its signal cost.

Notes
- Costs are compute-proportional (parameters x weighted tokens), not a vendor bill. The human is an idealised oracle with a fixed cost.
- Banking77 loading tries `mteb/banking77`, then `PolyAI/banking77`; check this in the smoke run.
