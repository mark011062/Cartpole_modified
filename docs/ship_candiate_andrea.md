Ship Candidate – Andrea Churchwell

Team: Team Two
Created: W19D4
Last Updated: 2026-02-13

------------------------------------------------------------

Selected Configuration

Algorithm: Tabular Q-Learning (CartPole-v1)

{
  "learning_rate": 0.351159658203541,
  "reward_shaping_scale": 0.1903872219557893,
  "num_bins": 9,
  "epsilon_decay_episodes": 1043,
  "discount_factor": 0.9863165063678174
}

------------------------------------------------------------

Enabled Improvements

- Adaptive learning rate
- Smart exploration
- Custom state discretization bins
- Reward shaping
- Optuna HPO (60 trials)

------------------------------------------------------------

Why This Config Was Selected

This configuration was identified through hyperparameter optimization
using Optuna combined with an extended episode budget (up to 1200 episodes).
Longer training revealed which configurations truly converged instead of
producing short high-reward spikes caused by variance.

Key observations during experimentation:

- Higher episode budgets reduced misleading early success signals.
- Multiple configurations reached strong rewards, but this one showed
  smoother learning progression and reasonable parameter values.
- Convergence occurred around episode ~966 during training.

This configuration:

- Reached maximum environment reward (500) during evaluation runs
- Demonstrated strong performance improvement over DEFAULT_PARAMS
- Avoided extreme or unstable hyperparameter ranges

------------------------------------------------------------

Performance Summary (Post W20 Multi-Seed Eval)

| Metric              | Baseline (A) | Ship Candidate (B) |
| ------------------- | ------------ | ------------------ |
| Mean Eval Return    | ~160         | ~306               |
| Episodes to Solve   | Not consistent | ~966 (best run) |
| Evaluation Episodes | 50           | 50                 |
| Stability           | High variance | Improved but unstable across seeds |

Important note:

CartPole has a reward cap of 500. Once solved, multiple configurations
can appear identical numerically. Convergence behavior and multi-seed
consistency were prioritized over single perfect scores.

------------------------------------------------------------

Evidence Summary

Artifacts saved:

- hpo_results/<timestamp>/best_params.json
- hpo_results/<timestamp>/all_trials.csv
- hpo_results/progress_report.html
- results/final_eval.csv

Best trial observed: #18

------------------------------------------------------------

Known Limitations

| Limitation            | Impact |
| --------------------- | ------ |
| RL seed variance      | Performance differs across seeds despite strong best-case reward |
| Discrete state space  | Limits generalization beyond CartPole |
| Reward cap saturation | Makes small improvements harder to detect numerically |

High scores alone are not treated as sufficient proof.

------------------------------------------------------------

Outcome (W20)

Multi-seed A/B evaluation was completed using `final_eval_qlearning.py`.

Results showed:

- Strong improvement over baseline
- Noticeable variance across seeds
- Some runs dropping below expected reward levels

Final Decision Status:

ITERATE

Further stability tuning or additional evaluation is recommended before
declaring this configuration a final ship candidate.

