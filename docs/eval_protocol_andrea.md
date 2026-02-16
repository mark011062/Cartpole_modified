Evaluation Protocol – Andrea Churchwell

Team: TeamTwo
Created: W19
Last Updated: 2026-02-13

---

Metric
| Field                | Value                                                          |
| -------------------- | -------------------------------------------------------------- |
| **Primary Metric**   | Mean evaluation episodic return                                |
| **How Measured**     | Average reward over fixed evaluation episodes (no exploration) |
| **Higher is Better** | Yes                                                            |

---

Seeds
| Field               | Value                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------- |
| **Number of Seeds** | 5 (Final W20 evaluation)                                                              |
| **Seed Strategy**   | Fixed seed list [0–4] applied equally to all variants                                 |       
| **Why This Many**   | Multi-seed evaluation used to reduce variance and validate HPO results in W20.        |

Clarification:
Single-seed results from W19 were exploratory only.  
Final A/B comparison uses multi-seed evaluation via `final_eval_qlearning.py`.

---

Training Budget
| Field                   | Value                                |
| ----------------------- | ------------------------------------ |
| **Episodes per Run**    | Up to 1200                           |
| **Evaluation Episodes** | 50                                   |
| **Environment Cap**     | 500 steps per episode                |

Rationale:
Higher episode counts improved convergence and stability for Q-Learning
with adaptive improvements enabled.

---

Stopping Rule
- Training stops after fixed episode budget
- No early stopping
- “Solved” logged when rolling average ≥195
- Final evaluation always performed after training

---

Comparison Rules
- Compare **mean evaluation return**, not single peaks
- Rank configurations by multi-seed evaluation mean
- Treat near-500 scores cautiously unless repeatable across seeds

When scores are close, prefer:
- smoother learning curves
- reasonable hyperparameters
- consistent convergence behavior

---

Known Sources of Variance
- Discretization granularity (num_bins)
- Reward shaping scale
- Exploration decay schedule
- Episode budget length
- RL stochasticity across seeds

These were explored via Optuna HPO in `w19d4_starter.py`.

---

Limitations
- High variance still present across seeds
- Some configurations achieve 500 but lack full stability
- Confidence interval analysis not yet formalized

---

W20 Updates
- Best HPO configuration evaluated across 5 seeds
- Baseline vs Ship Candidate compared using `results/final_eval.csv`
- Evaluation executed via scripted workflow instead of notebooks

---

Agreement

[ ] Full team sign-off pending

[x] Protocol reflects experiments actually run
[x] No mid-experiment metric changes were made
