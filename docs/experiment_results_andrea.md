# W20D4 Experiment Results (Andrea)

## Comparison
- A (Baseline): W19 baseline configuration
- B (Ship Candidate): HPO-tuned configuration from `hpo_results/.../best_params.json`

## Data
Final multi-seed evaluation results generated via `final_eval_qlearning.py`:
`results/final_eval.csv`

## Experiment Results

### A/B Comparison

Baseline (A) seed results:
- 91.88
- 101.86
- 237.64
- 142.90
- 226.92
Average ≈ 160

Ship Candidate (B) seed results:
- 500.00
- 226.62
- 239.40
- 117.52
- 449.04
Average ≈ 306

### Interpretation

The tuned configuration (B) improves average performance significantly compared to the baseline. However, performance varies across seeds, including one lower-performing run (~117 reward). This suggests the configuration is powerful but not fully stable.

### Decision

ITERATE.

The ship candidate shows strong improvement but requires additional evaluation across more seeds or stability adjustments before confidently shipping.
