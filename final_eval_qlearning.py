#!/usr/bin/env python3
import csv
import json
from pathlib import Path

from w19d4_starter import train_and_evaluate, DEFAULT_PARAMS

REPO = Path(__file__).resolve().parent
BEST_PARAMS_PATH = REPO / "hpo_results" / "andrea_churchwell_20260213_130938" / "best_params.json"
OUT_CSV_PATH = REPO / "results" / "final_eval.csv"

SEEDS = [0, 1, 2, 3, 4]

# We'll read switches from the JSON so it matches the run exactly
def load_best(path: Path):
    data = json.loads(path.read_text())
    params = data["best_params"]
    switches = data["switches"]

    # Ensure correct types
    params["num_bins"] = int(params["num_bins"])
    params["epsilon_decay_episodes"] = int(params["epsilon_decay_episodes"])
    return params, switches

def eval_variant(label: str, params: dict, switches: dict):
    rows = []
    for seed in SEEDS:
        res = train_and_evaluate(params=params, switches=switches, seed_offset=seed)
        mean_reward = float(res["eval_mean"])
        rows.append({"variant": label, "seed": seed, "mean_reward": mean_reward})
        print(f"{label} seed={seed} eval_mean={mean_reward:.2f}")
    return rows

def main():
    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing ship params: {BEST_PARAMS_PATH}")

    ship_params, ship_switches = load_best(BEST_PARAMS_PATH)

    # Baseline should use the SAME improvement switches for fairness,
    # but with DEFAULT_PARAMS (the "default" config).
    baseline_switches = ship_switches.copy()
    baseline_switches["USE_HPO"] = False  # not used by train_and_evaluate anyway

    print("\n=== Final Eval: A (DEFAULT_PARAMS) ===")
    rows_a = eval_variant("A", DEFAULT_PARAMS, baseline_switches)

    print("\n=== Final Eval: B (BEST_PARAMS) ===")
    rows_b = eval_variant("B", ship_params, baseline_switches)

    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "seed", "mean_reward"])
        w.writeheader()
        w.writerows(rows_a + rows_b)

    print(f"\n✅ Saved: {OUT_CSV_PATH}")

if __name__ == "__main__":
    main()