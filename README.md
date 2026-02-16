# W19 + W20 CartPole Mini Project

Train a reinforcement learning agent to balance a pole, then prove your improvements with evidence.

---

## What You're Building

A complete ML experimentation workflow:

```
Week 19: Train baseline → Learn PR workflow → Tune hyperparameters → Submit best config
Week 20: Package for reproducibility → Get reviewed → Run A/B test → Make final decision
```

---

## Your Team Roles

Assign these on Day 1 and write them in `docs/runbook.md`:

| Role                  | Who Does What                                |                      |
|-----------------------|----------------------------------------------|----------------------|
| **Runner** (1 person) | Runs the notebooks, exports CSV/JSON files   | Mark Young           |
| **Maintainer**        | Manages branches, merges PRs                 | Drashti Patel        |
| **Analyst**           | Writes experiment docs, interprets results   | Andrea Churchwell    |
| **Reviewer**          | Reviews teammate PRs, ensures quality        | Tashoy Miller        |

> **Note:** Everyone writes their own individual Google Doc reflection.

---

## Quick Start (Do This First)

### Using Google Colab (Recommended)

1. Click any `.ipynb` file in this repo
2. Click **"Open in Colab"** button (or File → Open in Colab)
3. Run cells top to bottom
4. Save your work: **File → Save a copy to GitHub**

### Using Local Python (Optional)

```bash
# Check Python version (must be 3.10+)
python --version

# Install packages
pip install gymnasium stable-baselines3 tensorboard optuna

# IMPORTANT: Run notebooks from the notebooks/ directory
cd notebooks
jupyter notebook W19D1_baseline.ipynb
```

---

# 📅 WEEK 19

---

## W19D1 — Train Your Baseline

**Goal:** Get a working PPO agent and agree on how you'll measure success.

### Step-by-Step

| Step | Action                                             | Where          |
|------|---------------------------------------------------------------------|
| 1    | Open `notebooks/W19D1_baseline.ipynb`              | GitHub → Colab |
| 2    | Run all cells (takes ~5-10 min)                    | Colab          |
| 3    | Write down your mean reward (e.g., "Mean: 245.3")  | Notebook output|
| 4    | Open `docs/eval_protocol.md` and fill in as a team | GitHub         |
| 5    | Open `docs/runbook.md` and assign team roles       | GitHub         |
| 6    | Post your baseline mean to Canvas                  | Canvas         |

### What You'll See

```
Training for 50,000 timesteps...
==================================================
BASELINE RESULTS
==================================================
Mean reward: 247.30 (+/- 15.42)
Eval episodes: 10
==================================================
```

### End of Day Checklist

- [x] Notebook runs without errors
- [x] Baseline mean reward recorded
- [x] `docs/eval_protocol.md` filled in (metric, seeds, timesteps)
- [x] `docs/runbook.md` filled in (Runner name, environment choice)
- [x] Posted to Canvas

---

## W19D2 — Learn the PR Workflow

**Goal:** Practice the branch → PR → review → merge cycle you'll use for real contributions.

### Step-by-Step

| Step| Action                                          | Where |
|-------------------------------------------------------|--------|
| 1   | Create a new branch: `feature/your-name-update` | GitHub |
| 2   | Edit any doc file (small change is fine)        | GitHub |
| 3   | Open a Pull Request to `main`                   | GitHub |
| 4   | Request a review from a teammate                | GitHub |
| 5   | Review another team's PR (leave 1 comment)      | GitHub |
| 6   | Fill in `docs/contribution_plan.md`             | GitHub |

### How to Create a Branch (GitHub Web)

1. Click the branch dropdown (says "main")
2. Type your new branch name
3. Click "Create branch: feature/..."

### How to Open a PR

1. After pushing changes, click "Compare & pull request"
2. Write a clear title and description
3. Click "Create pull request"
4. Click "Reviewers" → add a teammate

### End of Day Checklist

- [x] Created a feature branch
- [x] Opened a PR
- [x] Requested review from teammate
- [x] Reviewed another team's PR
- [x] `docs/contribution_plan.md` filled in

---

## W19D4 — Hyperparameter Optimization

**Goal:** Use Optuna to find better hyperparameters than the baseline.

### Step-by-Step

| Step | Action                                                | Where          |
|------|-------------------------------------------------------|----------------|
| 1    | Open `notebooks/W19D4_hpo.ipynb`                      | GitHub → Colab |
| 2    | Set `N_TRIALS = 10` (for in-class)                    | Colab          |
| 3    | Run all cells (~30-45 min)                            | Colab          |
| 4    | Look at the leaderboard output                        | Colab          |
| 5    | Download `hpo_leaderboard.csv` → commit to `results/` | GitHub         |
| 6    | Download `best_config.json` → commit to `results/`    | GitHub         |
| 7    | Fill in `docs/ship_candidate.md`                      | GitHub         |

### What You'll See

```
Top 10 Trials:
 trial  mean_reward  learning_rate  n_steps   gamma    ent_coef  clip_range
    7       487.20       0.000312      128  0.9901    0.000045        0.25
    3       421.50       0.000891       64  0.9856    0.000123        0.31
   ...
```

### End of Day Checklist

- [x] HPO notebook runs without errors
- [x] `results/hpo_leaderboard.csv` committed (10+ trials)
- [x] `results/best_config.json` committed
- [x] `docs/ship_candidate.md` filled in (why this config won)

---

## 📝 W19 Weekend Assignment (Due Sunday)

### Team Tasks

| Task              | Owner          | What to Do                                            |
|-------------------|----------------|-------------------------------------------------------|
| Expand HPO sweep  | **Runner**     | Change `N_TRIALS = 25`, run again, commit updated CSV |
| Finalize PR       | **Maintainer** | Make sure PR is merge-ready                           |
| Update docs       | **Analyst**    | Ensure all docs are complete                          |

### Individual Task (Everyone)

Write a **1-2 page Google Doc** answering:

1. What hyperparameters did you tune and why?
2. What does the evidence (CSV) show?
3. What's one limitation you observed?

Then add your link to `reports/individual_links.md`:

```markdown
| Your Name | [Your Google Doc](https://docs.google.com/...) |
```

### W19 Submission Checklist

- [x] `results/hpo_leaderboard.csv` has **25+ trials**
- [x] `results/best_config.json` exists
- [x] PR is merge-ready or merged
- [x] Your Google Doc link is in `reports/individual_links.md`

---

# 📅 WEEK 20

---

## W20D1 — Package Your Ship Candidate

**Goal:** Make your best config reproducible so reviewers can verify it works.

### Step-by-Step

| Step | Action | Where |
|------|--------|-------|
| 1 | Open HPO notebook and load `best_config.json` | Colab |
| 2 | Verify the config produces similar results | Colab |
| 3 | Update `docs/how_to_run.md` with exact steps | GitHub |
| 4 | Update `docs/runbook.md` with final instructions | GitHub |
| 5 | Open PR with any implementation changes | GitHub |

### What `docs/how_to_run.md` Should Include

```markdown
## How to Reproduce Our Results

1. Open `notebooks/W19D4_hpo.ipynb` in Colab
2. Run all cells
3. Check that `results/best_config.json` matches:
   - learning_rate: 0.000312
   - n_steps: 128
   - ...
4. Expected output: Mean reward ~450+
```

### End of Day Checklist

- [ ] Ship candidate config verified
- [ ] `docs/how_to_run.md` complete
- [ ] `docs/runbook.md` updated
- [ ] PR opened for implementation work

---

## W20D2 — PR Review Day

**Goal:** Get your PR reviewed, review others, and merge.

### Step-by-Step

| Step | Action | Where |
|------|--------|-------|
| 1 | Get assigned another team's PR to review | GitHub |
| 2 | Leave **2 inline comments** on specific lines | GitHub |
| 3 | Leave **1 summary comment** (approve or request changes) | GitHub |
| 4 | Check your own PR for review feedback | GitHub |
| 5 | Make fixes based on feedback, push new commits | GitHub |
| 6 | Once approved by instructor, merge your PR | GitHub |

### Good Review Comments

**Inline comment example:**
> "Line 45: Could you add a comment explaining why gamma=0.99 was chosen?"

**Summary comment example:**
> "Overall looks good! The run steps are clear. One suggestion: add the expected runtime. Approving with that minor note."

### End of Day Checklist

- [ ] Reviewed another team's PR (2 inline + 1 summary)
- [ ] Responded to feedback on your PR
- [ ] PR approved or changes requested addressed
- [ ] PR merged (or ready to merge)

---

## W20D4 — A/B Experiment

**Goal:** Compare your baseline (A) vs ship candidate (B) with real evidence.

### Step-by-Step

| Step | Action | Where |
|------|--------|-------|
| 1 | Fill in `docs/experiment_brief.md` (hypothesis, decision rule) | GitHub |
| 2 | **Runner:** Run baseline config with 5 different seeds | Colab |
| 3 | **Runner:** Run ship candidate with 5 different seeds | Colab |
| 4 | Export results to `results/final_eval.csv` | GitHub |
| 5 | Compute confidence interval for (B - A) | Colab |
| 6 | Team decides: **SHIP** / **ITERATE** / **REVERT** | Discussion |
| 7 | Fill in `docs/experiment_results.md` | GitHub |

### Understanding Your Results

| If CI for (B - A)... | Interpretation | Decision |
|----------------------|----------------|----------|
| Entirely above 0 | B is better | **SHIP** |
| Includes 0 | Can't tell | **ITERATE** (need more data) |
| Entirely below 0 | A is better | **REVERT** |

### End of Day Checklist

- [ ] `docs/experiment_brief.md` complete
- [ ] `results/final_eval.csv` has A and B results
- [ ] Confidence interval computed
- [ ] Decision made and documented
- [ ] `docs/experiment_results.md` complete

---

## 📝 W20 Weekend Assignment (Due Sunday)

### Team Tasks

| Task | Owner | What to Do |
|------|-------|------------|
| Finalize eval | **Runner** | Complete any remaining runs |
| Merge PR | **Maintainer** | Ensure PR is merged |
| Final docs | **Analyst** | All experiment docs complete |

### Individual Task (Everyone)

Write a **1-2 page Google Doc** answering:

1. What did your A/B comparison show?
2. What can you claim? What can't you claim? (uncertainty)
3. What's one experiment you'd run next?

Then add your link to `reports/individual_links.md`.

### W20 Submission Checklist

- [ ] `results/final_eval.csv` exists with A and B results
- [ ] `docs/experiment_brief.md` complete
- [ ] `docs/experiment_results.md` complete
- [ ] PR merged with approvals
- [ ] Your Google Doc link is in `reports/individual_links.md`

---

# 📁 Repository Structure

```
├── notebooks/
│   ├── W19D1_baseline.ipynb      ← Start here Day 1
│   └── W19D4_hpo.ipynb           ← Use this Day 4
│
├── docs/
│   ├── eval_protocol.md          ← Fill in W19D1
│   ├── runbook.md                ← Fill in W19D1, update throughout
│   ├── contribution_plan.md      ← Fill in W19D2
│   ├── ship_candidate.md         ← Fill in W19D4
│   ├── how_to_run.md             ← Fill in W20D1
│   ├── experiment_brief.md       ← Fill in W20D4
│   └── experiment_results.md     ← Fill in W20D4
│
├── results/
│   ├── hpo_leaderboard.csv       ← Export from HPO notebook
│   ├── best_config.json          ← Export from HPO notebook
│   └── final_eval.csv            ← Export from A/B experiment
│
├── reports/
│   └── individual_links.md       ← Add your Google Doc links here
│
└── README.md                      ← You are here
```

---

# 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Colab disconnects | Save frequently with File → Save to GitHub |
| "Module not found" | Run the install cell at the top again |
| Notebook won't open in Colab | Try: File → Open notebook → GitHub tab → paste repo URL |
| PR can't be merged | Check for merge conflicts, resolve them |
| Results look wrong | Check your seed matches the protocol |

---

# 📚 Resources

| Topic | Link |
|-------|------|
| CartPole environment | [gymnasium.farama.org](https://gymnasium.farama.org/environments/classic_control/cart_pole/) |
| PPO in SB3 | [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) |
| TensorBoard logging | [SB3 TensorBoard guide](https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html) |
| Optuna HPO | [optuna.org](https://optuna.org) |
| GitHub PRs | [GitHub PR docs](https://docs.github.com/articles/about-pull-requests) |

---

# ✅ Final Checklist

Before you're done with this project, make sure you have:

### Week 19
- [ ] Baseline trained and recorded
- [ ] Eval protocol documented
- [ ] Team roles assigned in runbook
- [ ] PR workflow practiced
- [ ] HPO sweep run (25+ trials)
- [ ] Ship candidate selected with rationale
- [ ] Individual Google Doc submitted

### Week 20
- [ ] Ship candidate packaged with run steps
- [ ] PR reviewed and merged
- [ ] A/B experiment completed
- [ ] Decision made (SHIP/ITERATE/REVERT)
- [ ] Uncertainty documented
- [ ] Individual Google Doc submitted

---


