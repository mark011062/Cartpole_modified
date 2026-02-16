# W20D4 Experiment Brief (Andrea)

## Goal
Compare baseline (A) vs ship candidate (B) using the same evaluation protocol.


## Variants
- **A (Baseline):** DEFAULT_PARAMS from `w19d4_starter.py`
- **B (Ship Candidate):** Best HPO configuration saved in:
  `hpo_results/<timestamp>/best_params.json`


## Primary Metric
Mean episodic return (average reward across evaluation episodes).

---

## Evaluation Plan
Final evaluation is performed using:
```
python final_eval_qlearning.py
```
This script:

- Loads `hpo_results/<timestamp>/best_params.json`
- Runs BOTH variants with identical improvement switches
- Uses 5 seeds per variant
- Saves results to: `results/final_eval.csv`
Columns:
- variant (A or B)
- seed
- mean_reward

---

## Decision Rule
Compute comparison of **B vs A**:

- If B consistently outperforms A → SHIP
- If results overlap heavily or are unstable → ITERATE
- If B performs worse → REVERT

(Note: Due to RL variance, interpretation focuses on overall trend and stability,
not just single runs.)

---

## Notes
- Evaluation settings match `docs/eval_protocol_andrea.md`.
- Notebook-based evaluation is no longer used; all experiments run through
`w19d4_starter.py` and `final_eval_qlearning.py`.