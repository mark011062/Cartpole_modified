# How to Reproduce Our Ship Candidate (Andrea)

## What this is
This document explains how to reproduce the final ship candidate using the
Q-Learning + HPO workflow implemented in `w19d4_starter.py`, along with the
saved best configuration located in the `hpo_results/` directory.

This reflects the FINAL W20 evaluation workflow and replaces the earlier
notebook-based process.

---

## Requirements
- Python 3.10+
- gymnasium
- numpy
- matplotlib
- optuna
- pandas

Dependencies are installed automatically when running `w19d4_starter.py`
because the script creates and manages its own virtual environment.

---

## Option A — Run HPO Training (Local Terminal)

From the repo root:

python w19d4_starter.py


This will:

- Create `.venv_hpo`
- Install dependencies automatically
- Run Optuna HPO trials
- Save results into:

hpo_results/<timestamp>/


Key artifact produced:

hpo_results/<timestamp>/best_params.json


Expected behavior:

- Best reward typically approaches ~500
- Terminal shows “Boss Level: HPO” summary
- Progress report updates in `hpo_results/progress_report.html`

---

## Option B — Run Final Evaluation (A/B Comparison)

To reproduce the final evaluation CSV used in W20:

python final_eval_qlearning.py


This script will:

- Load the saved `best_params.json`
- Evaluate Baseline (A) vs Ship Candidate (B)
- Run multiple seeded evaluations
- Write results to:

results/final_eval.csv


Expected outcome:

- Variant A average reward lower than Variant B
- Variant B shows stronger performance but some seed variance
- Variance is expected due to RL stochasticity

---

## Notes / Known Issues

- HPO results are stochastic and may differ slightly between runs.
- CartPole performance can vary across seeds; multi-seed evaluation is required.
- Notebook workflow (`W19D4_hpo.ipynb`) is NOT used for final evaluation anymore.