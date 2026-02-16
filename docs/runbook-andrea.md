## Runbook – Andrea Churchwell
### Overview

This runbook documents my understanding of our team’s progress during Week 19, including what we experimented with, what we learned, and what still needs to be formalized. The focus this week was primarily on experimentation and understanding behavior rather than completing a fully polished workflow.

### Monday – Baseline & Early Experiments

On Monday night, our focus shifted toward understanding what a “baseline” actually represents in practice.
- We ran baseline-style configurations and observed high variance in episodic return.
- At one point, we observed a run with a mean reward around ~258, which initially looked promising.
- After additional runs, it became clear that this was a lucky run rather than a stable improvement.
- This helped reinforce an important lesson: single runs are misleading, and averages across episodes and/or seeds matter more than peak values.

Key takeaway:
Baseline performance can fluctuate significantly, and strong-looking results must be validated through repeated evaluation.

### Tuesday – Team Repo, Branching, and Individual Improvements

Tuesday was focused on team coordination and exploring different improvement techniques.

Repo & Workflow
- We set up a shared team repository.
- Each team member worked on a separate branch to explore one type of improvement.
- While we did not complete a full PR → review → merge cycle for every change, the branching structure was established and used for experimentation.

Individual Contributions
- Andrea (me): Reward shaping
Explored modifying the reward signal to penalize poor pole angle and cart position.
- Drashti: State-related improvement
Focused on changes related to state representation / normalization.
- Tashoy: Exploration / tiebreaker logic
Worked on improving action selection when Q-values are similar.
- Mark: Learning rate strategies
- Investigated adaptive or adjusted learning rate behavior.

Key takeaway:
Different improvements target different failure modes, and no single switch guarantees success on its own.

### Friday – Hyperparameter Optimization (HPO)

Friday was focused on understanding and applying hyperparameter optimization using Optuna.

Learned how HPO systematically searches combinations of:
- learning rate
- discount factor
- epsilon decay
- reward shaping scale
- discretization bins

Observed that episodes and trials are critical:
- Too few episodes → agent doesn’t have time to learn.
- Too few trials → optimizer may miss strong configurations.

With sufficient trials and episodes, HPO was able to discover configurations that:
- Solved CartPole consistently
- Achieved high evaluation scores (up to 500)

Key takeaway:
HPO removes guesswork, but only works well when given enough time (episodes) and search budget (trials).

Current State of Results
We have strong evidence that:
- Baseline alone is unstable
- Individual improvements help but are inconsistent
- HPO produces the most reliable performance
- Results are currently directional rather than fully standardized.

A formal A/B comparison using identical seeds and evaluation settings is still needed.

## Reproducibility Notes

Experiments were run using the `w19d4_starter.py` script.
Final A/B evaluation was executed using `final_eval_qlearning.py`.

Hyperparameter optimization was performed using Optuna with varying numbers of trials
(25–120) and increased episode budgets to allow convergence.

Due to stochasticity in reinforcement learning, exact scores may vary between runs.
However, rerunning the script with similar episode counts and trials should reproduce
the overall performance trends observed (baseline < tuned < HPO).

Best-performing hyperparameters are saved in `hpo_results/*/best_params.json`.

Next Steps
- Standardize a true baseline (fixed params, fixed seeds)
- Select a single “ship candidate” from HPO results
- Run baseline vs candidate under identical evaluation protocol
- Formalize PR and merge workflow once experimentation stabilizes
- Document results in evaluation and results markdowns

Final Reflection

This week emphasized learning why certain configurations work rather than simply achieving a high score. The experiments clarified the importance of variance, evaluation methodology, and reproducibility, setting the stage for more rigorous comparisons next week.

---

## Week 20 Update (Andrea)

### Ship Candidate
Using the configuration stored in:
`hpo_results/<timestamp>/best_params.json`

(This reflects the final Q-Learning HPO workflow rather than the earlier notebook artifact.)

This config was selected from HPO results based on highest mean reward and stable evaluation performance.

### Reproducibility
Instructions to reproduce results are documented in:
`docs/how_to_run_andrea.md`

### Notes
Week 20 shifted away from notebook-based experimentation.
All final evaluation and reproducibility steps were executed through
scripted workflows (`w19d4_starter.py` and `final_eval_qlearning.py`)
to ensure consistent multi-seed comparison.