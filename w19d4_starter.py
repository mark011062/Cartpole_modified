#!/usr/bin/env python3
"""
W19D4: Q-Learning HPO with Optuna - Student Starter (FIXED)
===========================================================

Key fix:
- No more UnboundLocalError for `study`.
- We only reference `study` in HPO mode.

Also fixed:
- Correct "solved_at" checks (episode 0 is valid, so use `is not None`)
- Clearer mode logic (FIXED vs HPO)
- Save all_trials.csv only when HPO actually ran
"""

# =============================================================================
# SECTION 0: AUTO VENV SETUP
# =============================================================================

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, ".venv_hpo")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "hpo_results")
REQUIREMENTS = ["gymnasium", "numpy", "matplotlib", "optuna", "tqdm", "pandas"]


def is_in_venv():
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("HPO_VENV_ACTIVE") == "1"
    )


def setup_venv():
    print("=" * 60)
    print("Setting up virtual environment...")
    print("=" * 60)

    if not os.path.exists(VENV_DIR):
        print(f"Creating venv at {VENV_DIR}...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)

    if sys.platform == "win32":
        pip_path = os.path.join(VENV_DIR, "Scripts", "pip")
        python_path = os.path.join(VENV_DIR, "Scripts", "python")
    else:
        pip_path = os.path.join(VENV_DIR, "bin", "pip")
        python_path = os.path.join(VENV_DIR, "bin", "python")

    print("Installing dependencies...")
    # subprocess.run([pip_path, "install", "--quiet", "--upgrade", "pip"], check=True)
    # subprocess.run([pip_path, "install", "--quiet"] + REQUIREMENTS, check=True)
    subprocess.run([python_path, "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)
    subprocess.run([python_path, "-m", "pip", "install", "--quiet"] + REQUIREMENTS, check=True)
    print("Ready!\n")

    return python_path


def run_in_venv():
    python_path = setup_venv()
    env = os.environ.copy()
    env["HPO_VENV_ACTIVE"] = "1"
    args = [python_path, __file__] + sys.argv[1:]
    result = subprocess.run(args, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__" and not is_in_venv():
    run_in_venv()

# =============================================================================
# SECTION 1: IMPORTS (after venv is active)
# =============================================================================

import json
import time
import argparse
from datetime import datetime
from collections import defaultdict

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# SECTION 2: YOUR INFO
# =============================================================================

STUDENT_NAME = "Andrea Churchwell"  # <-- CHANGE THIS!

# =============================================================================
# SECTION 3: IMPROVEMENT SWITCHES
# =============================================================================

USE_ADAPTIVE_LR = True
USE_SMART_EXPLORATION = True
USE_CUSTOM_BINS = True
USE_REWARD_SHAPING = True
USE_HPO = True # Turn ON to run Optuna HPO

# =============================================================================
# SECTION 4: HPO CONFIGURATION
# =============================================================================

DEFAULT_N_TRIALS = 60

MAX_EPISODES = 1200
TARGET_REWARD = 195
ROLLING_WINDOW = 50

EVAL_EPISODES = 50
RANDOM_SEED = 42

# =============================================================================
# SECTION 5: DEFAULT & SEARCH SPACE
# =============================================================================
DEFAULT_PARAMS = {
    "learning_rate": 0.29819721079349215,
    "epsilon_decay_episodes": 930,
    "num_bins": 12,
    "reward_shaping_scale": 0.10592025516367046,
    "discount_factor": 0.9726239297186803,
}


def sample_hyperparameters(trial: optuna.Trial) -> dict:
    learning_rate = trial.suggest_float("learning_rate", 0.02, 0.4)
    reward_shaping_scale = trial.suggest_float("reward_shaping_scale", 0.01, 0.25)
    num_bins = trial.suggest_int("num_bins", 8, 20)
    epsilon_decay_episodes = trial.suggest_int("epsilon_decay_episodes", 200, 1200)
    discount_factor = trial.suggest_float("discount_factor", 0.95, 0.999)

    return {
        "learning_rate": learning_rate,
        "epsilon_decay_episodes": epsilon_decay_episodes,
        "num_bins": num_bins,
        "reward_shaping_scale": reward_shaping_scale,
        "discount_factor": discount_factor,
    }


# =============================================================================
# SECTION 6: Q-LEARNING AGENT
# =============================================================================


class QLearningAgent:
    def __init__(self, params: dict, switches: dict):
        self.learning_rate = params["learning_rate"]
        self.discount_factor = params["discount_factor"]
        self.epsilon_decay_episodes = params["epsilon_decay_episodes"]
        self.num_bins = params["num_bins"]
        self.reward_shaping_scale = params["reward_shaping_scale"]

        self.switches = switches

        self.epsilon = 1.0
        self.epsilon_end = 0.0

        self.q_table = defaultdict(lambda: np.zeros(2))
        self.bins = self.create_bins()

    def create_bins(self):
        n = self.num_bins

        if self.switches["USE_CUSTOM_BINS"]:
            cart_pos_bins = np.linspace(-2.4, 2.4, max(2, n - 2))
            cart_vel_bins = np.linspace(-3.0, 3.0, max(2, n - 2))
            pole_angle_bins = np.linspace(-0.25, 0.25, n + 4)
            pole_vel_bins = np.linspace(-3.5, 3.5, n + 2)
        else:
            cart_pos_bins = np.linspace(-2.4, 2.4, n)
            cart_vel_bins = np.linspace(-3.0, 3.0, n)
            pole_angle_bins = np.linspace(-0.25, 0.25, n)
            pole_vel_bins = np.linspace(-3.5, 3.5, n)

        return {
            "cart_pos": cart_pos_bins,
            "cart_vel": cart_vel_bins,
            "pole_angle": pole_angle_bins,
            "pole_vel": pole_vel_bins,
        }

    def discretize(self, state):
        cart_pos, cart_vel, pole_angle, pole_vel = state
        return (
            int(np.digitize(cart_pos, self.bins["cart_pos"])),
            int(np.digitize(cart_vel, self.bins["cart_vel"])),
            int(np.digitize(pole_angle, self.bins["pole_angle"])),
            int(np.digitize(pole_vel, self.bins["pole_vel"])),
        )

    def get_learning_rate(self, episode: int) -> float:
        if self.switches["USE_ADAPTIVE_LR"]:
            decay = 1.0 / (1.0 + 0.001 * episode)
            return self.learning_rate * decay
        return self.learning_rate

    def select_action(self, state, training: bool = True) -> int:
        discrete_state = self.discretize(state)
        q_values = self.q_table[discrete_state]

        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, 2)

        if q_values[0] == q_values[1]:
            return np.random.randint(0, 2)

        return int(np.argmax(q_values))

    def decay_epsilon(self, episode: int):
        if self.switches["USE_SMART_EXPLORATION"]:
            if episode < self.epsilon_decay_episodes:
                self.epsilon = 1.0 - (episode / self.epsilon_decay_episodes)
            else:
                self.epsilon = self.epsilon_end
        else:
            self.epsilon = max(0.01, self.epsilon * 0.995)

    def shape_reward(self, reward: float, state, done: bool) -> float:
        if self.switches["USE_REWARD_SHAPING"] and self.reward_shaping_scale > 0:
            cart_pos, cart_vel, pole_angle, pole_vel = state
            angle_penalty = abs(pole_angle) * self.reward_shaping_scale
            edge_penalty = 0.0
            if abs(cart_pos) > 1.5:
                edge_penalty = (abs(cart_pos) - 1.5) * 0.1
            return reward - angle_penalty - edge_penalty
        return reward

    def update(self, state, action, reward, next_state, done: bool, episode: int):
        ds = self.discretize(state)
        nds = self.discretize(next_state)

        shaped_reward = self.shape_reward(reward, state, done)
        lr = self.get_learning_rate(episode)

        old_value = self.q_table[ds][action]
        if done:
            td_target = shaped_reward
        else:
            td_target = shaped_reward + self.discount_factor * np.max(self.q_table[nds])

        self.q_table[ds][action] = old_value + lr * (td_target - old_value)


# =============================================================================
# SECTION 7: TRAINING FUNCTION
# =============================================================================


def train_and_evaluate(params: dict, switches: dict, seed_offset: int = 0) -> dict:
    np.random.seed(RANDOM_SEED + seed_offset)

    agent = QLearningAgent(params, switches)
    env = gym.make("CartPole-v1")

    episode_rewards = []
    solved_at = None

    for episode in range(MAX_EPISODES):
        state, _ = env.reset(seed=RANDOM_SEED + seed_offset + episode)
        episode_reward = 0.0

        for _step in range(500):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            agent.update(state, action, reward, next_state, done, episode)
            state = next_state
            episode_reward += float(reward)

            if done:
                break

        episode_rewards.append(episode_reward)
        agent.decay_epsilon(episode)

        if len(episode_rewards) >= ROLLING_WINDOW:
            rolling_avg = float(np.mean(episode_rewards[-ROLLING_WINDOW:]))
            if rolling_avg >= TARGET_REWARD and solved_at is None:
                solved_at = episode
                break

    # Evaluation (no exploration)
    eval_scores = []
    for ep in range(EVAL_EPISODES):
        state, _ = env.reset(seed=1000 + ep)
        total_reward = 0.0

        for _step in range(500):
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        eval_scores.append(total_reward)

    env.close()

    return {
        "eval_mean": float(np.mean(eval_scores)),
        "eval_std": float(np.std(eval_scores)),
        "episode_rewards": episode_rewards,
        "solved_at": solved_at,
        "total_episodes": len(episode_rewards),
    }


# =============================================================================
# SECTION 8: OPTUNA OBJECTIVE
# =============================================================================


def create_objective(switches: dict, dashboard=None, training_histories=None):
    def objective(trial: optuna.Trial):
        params = sample_hyperparameters(trial)
        result = train_and_evaluate(params, switches, seed_offset=trial.number)

        if training_histories is not None:
            training_histories[trial.number] = {
                "params": params,
                "episode_rewards": result["episode_rewards"],
                "eval_mean": result["eval_mean"],
                "solved_at": result["solved_at"],
            }

        if dashboard is not None:
            dashboard.update_trial(trial.number, result["eval_mean"], params)

        return result["eval_mean"]

    return objective


# =============================================================================
# SECTION 9: LIVE DASHBOARD
# =============================================================================


class HPODashboard:
    def __init__(self, n_trials: int):
        self.n_trials = n_trials
        self.trial_rewards = []
        self.best_rewards = []
        self.trial_params = []

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 9))
        self.fig.suptitle(
            f"Q-Learning HPO Dashboard - {STUDENT_NAME}",
            fontsize=12,
            fontweight="bold",
        )

        self.ax1 = self.axes[0, 0]
        (self.line_best,) = self.ax1.plot([], [], "g-", linewidth=2, label="Best So Far")
        (self.line_trial,) = self.ax1.plot([], [], "bo", alpha=0.5, label="Trial Reward")
        self.ax1.set_xlabel("Trial")
        self.ax1.set_ylabel("Mean Reward")
        self.ax1.set_xlim(0, n_trials)
        self.ax1.set_ylim(0, 550)
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title("HPO Progress")

        self.ax2 = self.axes[0, 1]
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        self.ax2.set_title("Improvement Switches")
        self._draw_switches()

        self.ax3 = self.axes[1, 0]
        self.ax3.set_xlim(0, 1)
        self.ax3.set_ylim(0, 1)
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        self.ax3.set_title("Current Trial Parameters")
        self.params_text = self.ax3.text(
            0.1,
            0.9,
            "Waiting for first trial...",
            fontfamily="monospace",
            fontsize=10,
            verticalalignment="top",
        )

        self.ax4 = self.axes[1, 1]
        self.ax4.set_facecolor("#1a1a2e")
        self.ax4.set_xlim(0, 1)
        self.ax4.set_ylim(0, 1)
        self.ax4.set_xticks([])
        self.ax4.set_yticks([])
        self.ax4.set_title("Trial Log")
        self.log_text = self.ax4.text(
            0.02,
            0.98,
            "",
            fontfamily="monospace",
            fontsize=8,
            color="#00ff00",
            verticalalignment="top",
        )
        self.log_lines = []

        plt.tight_layout()
        plt.pause(0.1)

    def _draw_switches(self):
        switches = [
            ("Adaptive LR", USE_ADAPTIVE_LR),
            ("Smart Exploration", USE_SMART_EXPLORATION),
            ("Custom Bins", USE_CUSTOM_BINS),
            ("Reward Shaping", USE_REWARD_SHAPING),
        ]
        for i, (name, enabled) in enumerate(switches):
            y = 0.8 - i * 0.2
            color = "#10b981" if enabled else "#6b7280"
            symbol = "✓" if enabled else "✗"
            self.ax2.text(
                0.1,
                y,
                f"{symbol} {name}",
                fontsize=12,
                color=color,
                fontweight="bold" if enabled else "normal",
            )

    def update_trial(self, trial_num: int, reward: float, params: dict):
        self.trial_rewards.append(float(reward))
        if not self.best_rewards:
            self.best_rewards.append(float(reward))
        else:
            self.best_rewards.append(max(self.best_rewards[-1], float(reward)))

        self.trial_params.append(params)

        trials = list(range(1, len(self.trial_rewards) + 1))
        self.line_trial.set_data(trials, self.trial_rewards)
        self.line_best.set_data(trials, self.best_rewards)

        params_str = f"Trial {trial_num + 1} / {self.n_trials}\n"
        params_str += f"Reward: {reward:.1f}\n\n"
        params_str += "Parameters:\n"
        for key, value in params.items():
            if isinstance(value, float):
                params_str += f"  {key}: {value:.4f}\n"
            else:
                params_str += f"  {key}: {value}\n"
        self.params_text.set_text(params_str)

        log_line = f"Trial {trial_num + 1}: reward={reward:.1f}"
        self.log_lines.append(log_line)
        if len(self.log_lines) > 15:
            self.log_lines.pop(0)
        self.log_text.set_text("\n".join(self.log_lines))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def close(self):
        plt.ioff()
        plt.close()


# =============================================================================
# SECTION 10: PROGRESS + REPORT (UNCHANGED HELPERS)
# =============================================================================


def get_config_label(switches: dict) -> str:
    on_switches = []
    if switches["USE_ADAPTIVE_LR"]:
        on_switches.append("Adaptive LR")
    if switches["USE_SMART_EXPLORATION"]:
        on_switches.append("Smart Exploration")
    if switches["USE_CUSTOM_BINS"]:
        on_switches.append("Custom Bins")
    if switches["USE_REWARD_SHAPING"]:
        on_switches.append("Reward Shaping")

    on_count = len(on_switches)
    if on_count == 0:
        return "Baseline"
    if on_count == 1:
        return on_switches[0]
    if on_count == 4:
        return "All Improvements"
    return " + ".join(on_switches)


def detect_level(switches: dict, n_trials: int) -> str:
    on_count = sum(
        [
            switches["USE_ADAPTIVE_LR"],
            switches["USE_SMART_EXPLORATION"],
            switches["USE_CUSTOM_BINS"],
            switches["USE_REWARD_SHAPING"],
        ]
    )
    if on_count == 0:
        return "Level 1: Baseline"
    if on_count < 4:
        return "Level 2: Some Improvements"
    if n_trials <= 5:
        return "Level 3: Full Power"
    return "Boss Level: HPO"


def compute_rolling_avg(rewards, window=20):
    rolling = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        rolling.append(sum(rewards[start : i + 1]) / (i - start + 1))
    return rolling


def save_to_progress_history(run_data: dict) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    history_file = os.path.join(RESULTS_DIR, "progress_history.json")

    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = {"student_name": STUDENT_NAME, "runs": []}

    history["runs"].append(run_data)
    history["student_name"] = STUDENT_NAME

    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    return history


def generate_progress_report(history: dict) -> str:
    # (Keeping your original HTML generator exactly as-is, to avoid accidental breakage.)
    # If you want, I can also refactor it later, but this keeps behavior stable.
    # NOTE: This is identical to your previous version except for being placed here.
    runs = history.get("runs", [])
    student_name = history.get("student_name", "Student")

    best_run = max(runs, key=lambda r: r.get("best_reward", 0)) if runs else None
    best_reward = best_run["best_reward"] if best_run else 0

    baseline_runs = [r for r in runs if r.get("config_label") == "Baseline"]
    baseline_reward = baseline_runs[0]["best_reward"] if baseline_runs else None

    runs_with_curves = [r for r in runs if r.get("rolling_avg")]

    config_colors = {
        "Baseline": "#94a3b8",
        "Adaptive LR": "#f59e0b",
        "Smart Exploration": "#3b82f6",
        "Custom Bins": "#8b5cf6",
        "Reward Shaping": "#ec4899",
        "All Improvements": "#10b981",
    }

    runs_json = json.dumps(runs)

    learning_curve_datasets = []
    for run in runs_with_curves:
        label = run.get("config_label", "Unknown")
        color = config_colors.get(label, "#6366f1")
        rolling_avg = run.get("rolling_avg", [])
        if rolling_avg:
            learning_curve_datasets.append(
                {
                    "label": label,
                    "data": rolling_avg,
                    "borderColor": color,
                    "backgroundColor": "transparent",
                    "borderWidth": 2,
                    "tension": 0.3,
                    "pointRadius": 0,
                }
            )

    datasets_json = json.dumps(learning_curve_datasets)

    convergence_data = []
    for run in runs:
        label = run.get("config_label", "Unknown")
        episodes = run.get("episodes_to_195")
        if episodes is not None:
            convergence_data.append({"label": label, "episodes": episodes})
    convergence_json = json.dumps(convergence_data)

    fastest = min(convergence_data, key=lambda x: x["episodes"]) if convergence_data else None
    fastest_label = fastest["label"] if fastest else "N/A"
    fastest_episodes = fastest["episodes"] if fastest else "N/A"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Training Metrics - {student_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
/* (style unchanged) */
:root {{
  --primary: #6366f1;
  --secondary: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --bg: #0f172a;
  --bg-card: #1e293b;
  --text: #f1f5f9;
  --text-muted: #94a3b8;
  --border: #334155;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  padding: 2rem;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}
header {{
  text-align: center;
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 2px solid var(--primary);
}}
header h1 {{
  font-size: 2rem;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}
.stats-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}}
.stat-card {{
  background: var(--bg-card);
  border-radius: 12px;
  padding: 1.25rem;
  text-align: center;
  border: 1px solid var(--border);
}}
.stat-card.best {{
  border-color: var(--secondary);
  background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(99,102,241,0.1));
}}
.stat-value {{
  font-size: 1.75rem;
  font-weight: bold;
  color: var(--secondary);
}}
.stat-label {{
  color: var(--text-muted);
  font-size: 0.85rem;
}}
.chart-container {{
  background: var(--bg-card);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  border: 1px solid var(--border);
}}
.chart-container h3 {{
  margin-bottom: 1rem;
  color: var(--primary);
}}
.chart-container p {{
  color: var(--text-muted);
  font-size: 0.9rem;
  margin-bottom: 1rem;
}}
.chart-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 1rem;
}}
th, td {{
  padding: 0.75rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
}}
th {{ color: var(--primary); }}
tr:hover {{ background: rgba(99,102,241,0.1); }}
.config-badge {{
  display: inline-block;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 500;
  background: var(--primary);
  color: white;
}}
.best-row {{ background: rgba(16,185,129,0.2) !important; }}
.switch-on {{ color: var(--secondary); }}
.switch-off {{ color: var(--text-muted); }}
.no-data {{
  text-align: center;
  padding: 2rem;
  color: var(--text-muted);
}}
@media (max-width: 768px) {{
  .chart-row {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="container">
<header>
  <h1>Q-Learning Training Metrics</h1>
  <p>{student_name} | {len(runs)} configurations tested</p>
</header>

<div class="stats-grid">
  <div class="stat-card best">
    <div class="stat-value">{best_reward:.1f}</div>
    <div class="stat-label">Best Eval Score</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{f"{baseline_reward:.1f}" if baseline_reward is not None else "N/A"}</div>
    <div class="stat-label">Baseline Score</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{fastest_label}</div>
    <div class="stat-label">Fastest Convergence</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{fastest_episodes if fastest_episodes != "N/A" else "N/A"}</div>
    <div class="stat-label">Episodes to 195</div>
  </div>
</div>

<div class="chart-container">
  <h3>Learning Curves Comparison</h3>
  <p>Rolling average reward (window=20) during training. Higher and faster = better learning.</p>
  <canvas id="learningCurvesChart" height="120"></canvas>
</div>

<div class="chart-row">
  <div class="chart-container">
    <h3>Convergence Speed</h3>
    <p>Episodes needed to reach 195+ reward (CartPole "solved")</p>
    <canvas id="convergenceChart"></canvas>
  </div>
  <div class="chart-container">
    <h3>Final Evaluation Scores</h3>
    <p>Mean reward over 50 evaluation episodes</p>
    <canvas id="scoresChart"></canvas>
  </div>
</div>

<div class="chart-container">
  <h3>Configuration Comparison</h3>
  <table>
    <thead>
      <tr>
        <th>Configuration</th>
        <th>Switches</th>
        <th>Eval Score</th>
        <th>Episodes to 195</th>
        <th>Improvement</th>
      </tr>
    </thead>
    <tbody id="comparisonTable"></tbody>
  </table>
</div>

</div>

<script>
const runs = {runs_json};
const datasets = {datasets_json};
const convergenceData = {convergence_json};
const bestReward = {best_reward};
const baselineReward = {baseline_reward if baseline_reward is not None else 'null'};

// Learning curves
const lcCanvas = document.getElementById('learningCurvesChart');
if (datasets.length > 0) {{
  const maxLen = Math.max(...datasets.map(d => d.data.length));
  const labels = Array.from({{length: maxLen}}, (_, i) => i + 1);

  new Chart(lcCanvas.getContext('2d'), {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      interaction: {{ intersect: false, mode: 'index' }},
      plugins: {{
        legend: {{ position: 'top', labels: {{ color: '#f1f5f9' }} }}
      }},
      scales: {{
        y: {{
          beginAtZero: true,
          max: 550,
          grid: {{ color: '#334155' }},
          ticks: {{ color: '#94a3b8' }}
        }},
        x: {{
          grid: {{ color: '#334155' }},
          ticks: {{ color: '#94a3b8' }}
        }}
      }}
    }}
  }});
}} else {{
  lcCanvas.parentElement.innerHTML += '<p class="no-data">Run experiments to see learning curves</p>';
}}

// Convergence
const convCanvas = document.getElementById('convergenceChart');
if (convergenceData.length > 0) {{
  new Chart(convCanvas.getContext('2d'), {{
    type: 'bar',
    data: {{
      labels: convergenceData.map(d => d.label),
      datasets: [{{
        label: 'Episodes to 195',
        data: convergenceData.map(d => d.episodes)
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }}
    }}
  }});
}} else {{
  convCanvas.parentElement.innerHTML += '<p class="no-data">No runs reached 195 reward yet</p>';
}}

// Scores
const scoresCanvas = document.getElementById('scoresChart');
const scoreData = runs.map(r => ({{ label: r.config_label || 'Unknown', score: r.best_reward }}));
new Chart(scoresCanvas.getContext('2d'), {{
  type: 'bar',
  data: {{
    labels: scoreData.map(d => d.label),
    datasets: [{{
      label: 'Eval Score',
      data: scoreData.map(d => d.score)
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }}
  }}
}});

// Table
const tbody = document.getElementById('comparisonTable');
runs.forEach(run => {{
  const switches = run.switches || {{}};
  const switchStr = [
    switches.USE_ADAPTIVE_LR ? '<span class="switch-on">LR</span>' : '<span class="switch-off">LR</span>',
    switches.USE_SMART_EXPLORATION ? '<span class="switch-on">Exp</span>' : '<span class="switch-off">Exp</span>',
    switches.USE_CUSTOM_BINS ? '<span class="switch-on">Bins</span>' : '<span class="switch-off">Bins</span>',
    switches.USE_REWARD_SHAPING ? '<span class="switch-on">Rew</span>' : '<span class="switch-off">Rew</span>'
  ].join(' ');

  const improvement = baselineReward ? ((run.best_reward / baselineReward - 1) * 100).toFixed(0) : 'N/A';
  const improvementStr = improvement !== 'N/A' ? (improvement > 0 ? '+' + improvement + '%' : improvement + '%') : 'N/A';

  tbody.innerHTML += `
    <tr class="${{run.best_reward === bestReward ? 'best-row' : ''}}">
      <td><span class="config-badge">${{run.config_label || 'Unknown'}}</span></td>
      <td>${{switchStr}}</td>
      <td>${{run.best_reward.toFixed(1)}}</td>
      <td>${{run.episodes_to_195 !== null ? run.episodes_to_195 : 'N/A'}}</td>
      <td>${{improvementStr}}</td>
    </tr>
  `;
}});
</script>
</body>
</html>"""

    report_path = os.path.join(RESULTS_DIR, "progress_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return report_path


# =============================================================================
# SECTION 11: MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="W19D4: Q-Learning HPO with Optuna")
    parser.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--fixed", action="store_true", help="Force fixed-mode (no HPO)")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = STUDENT_NAME.replace(" ", "_").lower()
    run_dir = os.path.join(RESULTS_DIR, f"{safe_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    switches = {
        "USE_ADAPTIVE_LR": USE_ADAPTIVE_LR,
        "USE_SMART_EXPLORATION": USE_SMART_EXPLORATION,
        "USE_CUSTOM_BINS": USE_CUSTOM_BINS,
        "USE_REWARD_SHAPING": USE_REWARD_SHAPING,
        "USE_HPO": USE_HPO,
    }

    # Decide mode ONCE, and use it everywhere consistently
    run_hpo = bool(USE_HPO and (not args.fixed))
    n_trials_effective = int(args.trials) if run_hpo else 1

    print("=" * 60)
    print("  W19D4: Q-Learning HPO")
    print(f"  Student: {STUDENT_NAME}")
    print("=" * 60)
    print("\n  Improvement Switches:")
    print(f"    {'✓' if USE_ADAPTIVE_LR else '✗'} Adaptive Learning Rate")
    print(f"    {'✓' if USE_SMART_EXPLORATION else '✗'} Smart Exploration")
    print(f"    {'✓' if USE_CUSTOM_BINS else '✗'} Custom Bins")
    print(f"    {'✓' if USE_REWARD_SHAPING else '✗'} Reward Shaping")
    print(f"    {'✓' if USE_HPO else '✗'} HPO (Hyperparameter Optimization)")
    print("\n  Training Configuration:")
    print(f"    Max episodes: {MAX_EPISODES}")
    print(f"    Target (solved): {TARGET_REWARD} rolling avg")
    print(f"    Eval episodes: {EVAL_EPISODES}")
    print(f"    Mode: {'HPO' if run_hpo else 'FIXED'}")
    if run_hpo:
        print(f"    HPO trials: {args.trials}")
    print("=" * 60)

    start_time = time.time()

    study = None  # ALWAYS define; only used when run_hpo=True

    if not run_hpo:
        print(f"\nRunning with FIXED default parameters (no HPO)...")
        # result = train_and_evaluate(DEFAULT_PARAMS, switches, seed_offset=0)
        result = train_and_evaluate(DEFAULT_PARAMS, switches, seed_offset=args.seed_offset)
        best_reward = result["eval_mean"]
        best_params_dict = DEFAULT_PARAMS.copy()
        episode_rewards = result["episode_rewards"]
        solved_at = result["solved_at"]
    else:
        dashboard = None
        if not args.no_plot:
            dashboard = HPODashboard(args.trials)

        training_histories = {}

        study = optuna.create_study(
            study_name=f"qlearning_hpo_{timestamp}",
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_SEED),
        )

        print(f"\nStarting HPO with {args.trials} trials...")
        study.optimize(
            create_objective(switches, dashboard, training_histories),
            n_trials=args.trials,
            show_progress_bar=True,
        )

        if dashboard is not None:
            dashboard.close()

        best_reward = float(study.best_trial.value)
        best_params_dict = {
            k: float(v) if isinstance(v, float) else v for k, v in study.best_trial.params.items()
        }
        best_trial_num = int(study.best_trial.number)
        best_training = training_histories.get(best_trial_num, {})
        episode_rewards = best_training.get("episode_rewards", [])
        solved_at = best_training.get("solved_at")

    total_time = time.time() - start_time

    # Save best params
    best_params_payload = {
        "student_name": STUDENT_NAME,
        "best_reward": float(best_reward),
        "best_params": best_params_dict,
        "switches": switches,
        "n_trials": n_trials_effective,
        "total_time_seconds": float(total_time),
        "timestamp": timestamp,
    }
    with open(os.path.join(run_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params_payload, f, indent=2)

    # Save all trials CSV ONLY if HPO actually ran
    if run_hpo and study is not None:
        df = study.trials_dataframe()
        df.to_csv(os.path.join(run_dir, "all_trials.csv"), index=False)

    rolling_avg = compute_rolling_avg(episode_rewards, window=20) if episode_rewards else []
    config_label = get_config_label(switches)
    if run_hpo and args.trials > 1:
        config_label += " + HPO"

    level = detect_level(switches, n_trials_effective)

    run_data = {
        "timestamp": timestamp,
        "level": level,
        "config_label": config_label,
        "switches": switches,
        "n_trials": n_trials_effective,
        "best_reward": float(best_reward),
        "best_params": best_params_dict,
        "training_curve": episode_rewards,
        "rolling_avg": rolling_avg,
        "episodes_to_195": solved_at,
        "total_episodes": len(episode_rewards),
    }
    history = save_to_progress_history(run_data)
    report_path = generate_progress_report(history)

    print("\n" + "=" * 60)
    print(f"  {level}")
    print("=" * 60)
    print(f"  Best Reward: {best_reward:.1f}")
    print(f"  Episodes Trained: {len(episode_rewards)}")
    if solved_at is not None:
        print(f"  Solved at Episode: {solved_at} ✓")
    else:
        print("  Solved: No")

    if run_hpo and study is not None:
        print("\n  Top Trials:")
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
        for i, trial in enumerate(sorted_trials[:5]):
            if trial.value is not None:
                print(f"    {i+1}. Trial #{trial.number}: {trial.value:.1f}")

    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"\n  Time: {minutes}m {seconds}s")

    print("\n" + "-" * 60)
    print(f"  PROGRESS: {len(history['runs'])} runs completed")
    print("-" * 60)

    print("\n" + "=" * 60)
    print("  VIEW YOUR PROGRESS")
    print("=" * 60)
    print(f"  Open: {report_path}")
    print("  (Interactive charts showing all your runs!)")
    print("=" * 60)


if __name__ == "__main__":
    main()