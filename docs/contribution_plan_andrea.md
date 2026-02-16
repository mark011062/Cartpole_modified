Contribution Plan – Andrea Churchwell

Team: TeamTwo
Created: W19D2
Last Updated: 2026-02-13

Personal Focus

My focus throughout this project was understanding why CartPole performance changes, not just how to increase the reward number. I concentrated on hyperparameter tuning, result interpretation, and connecting observed performance back to training dynamics such as episode budgets, trial counts, and variance across runs.

Rather than chasing a single high score, my goal was to build a workflow that could distinguish stable learning from lucky outcomes.

Problem Context

Early experiments showed that individual runs could produce deceptively high rewards. Some configurations appeared strong at first but failed to repeat consistently.

The main problem I focused on was learning how to:

Identify when a result is noise versus genuine improvement

Use systematic hyperparameter optimization (HPO) to reduce guesswork

Evaluate performance using structured comparisons instead of isolated runs

Scope of My Work
What I DID

Ran multiple CartPole experiments to understand baseline variance

Documented how “lucky” runs can inflate perceived performance

Used Optuna to perform structured hyperparameter optimization

Increased episode budgets and trial counts to allow learning to converge

Interpreted HPO outputs to identify reliable configurations

Saved best-performing configurations and trial artifacts for reproducibility

Executed a multi-seed A/B evaluation using final_eval_qlearning.py

Generated results/final_eval.csv to compare baseline vs tuned configuration

What I DID NOT Do

Modify the core algorithm architecture

Introduce entirely new RL algorithms

Claim baseline improvements without validation

Finalize team-wide merge or PR workflow beyond experimentation

| File                               | Description                  |
| ---------------------------------- | ---------------------------- |
| `w19d4_starter.py`                 | Main training + HPO workflow |
| `final_eval_qlearning.py`          | Multi-seed evaluation script |
| `hpo_results/*/best_params.json`   | Saved best configurations    |
| `hpo_results/*/all_trials.csv`     | Optuna trial distributions   |
| `hpo_results/progress_report.html` | Visual progress tracking     |
| `results/final_eval.csv`           | Final A/B evaluation output  |


Best HPO configuration achieving strong evaluation performance

Optuna trial logs showing performance spread and search behavior

Progress report visualizing multiple experimental runs

Multi-seed evaluation results comparing baseline vs ship candidate

These artifacts support conclusions about stability and variance, rather than relying on a single peak reward.

Key Learnings

High rewards early in training can be misleading without repetition.

Increasing episode budgets allows the agent to converge more reliably.

Increasing HPO trials helps discover better regions of the parameter space.

HPO works best when paired with sufficient training time.

Solving CartPole (≥195) does not automatically imply stability.

Multi-seed evaluation is essential for judging real improvement.

Current Limitations

Variance remains high across seeds even with tuned parameters.

Some evaluation runs still drop significantly despite strong averages.

Formal statistical confidence intervals were not implemented.

Team workflow (PR review/merge process) is still developing.

Next Steps

Investigate stability improvements (exploration schedule, reward shaping tuning).

Expand evaluation across additional seeds if time permits.

Improve clarity of evaluation reporting and decision criteria.

Continue refining reproducibility and documentation practices.

Reflection

This project shifted my perspective from “getting a high score” to understanding what that score actually represents. The combination of HPO experimentation and multi-seed evaluation helped clarify how reinforcement learning results can vary, and why structured evaluation matters.

The work completed here builds a foundation for more rigorous experimentation moving forward, with a stronger emphasis on reproducibility, stability, and honest interpretation of results.