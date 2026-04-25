"""
evaluation/aggregate_results.py
================================
Loads all judge and metric logs, aggregates them into the three-checkpoint
comparison table from the assignment, and prints/saves the final summary.

Run this after all evaluation scripts have completed:
    python evaluation/aggregate_results.py --config config.yaml

Outputs:
    logs/final_summary.json
    logs/forgetting_analysis.json
"""

import argparse
import json
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_json_safe(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    print(f"  [missing] {path}")
    return {}


def fmt(val, pct=False, decimals=3):
    if val is None:
        return "N/A"
    if pct:
        return f"{val:.1%}"
    return f"{val:.{decimals}f}"


def build_table(summary: dict) -> str:
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("THREE-CHECKPOINT COMPARISON TABLE")
    lines.append("=" * 100)

    header = (
        f"{'Checkpoint':<30} {'Alpaca Win%':>12} {'ROUGE-L':>8} "
        f"{'BERTScore':>10} {'JSON Valid%':>12} {'Schema%':>8} {'ExactMatch%':>12}"
    )
    lines.append(header)
    lines.append("-" * 100)

    for ckpt_id, label in [(0, "Ckpt 0: Untuned Base"),
                            (1, "Ckpt 1: After Stage 1 (Alpaca)"),
                            (2, "Ckpt 2: After Stage 2 (JSON)")]:
        ckpt_data = summary.get(f"checkpoint_{ckpt_id}", {})
        alpaca_win = ckpt_data.get("alpaca_win_rate_vs_baseline")
        rougeL     = ckpt_data.get("alpaca_rougeL")
        bert_f1    = ckpt_data.get("alpaca_bertscore_f1")
        json_val   = ckpt_data.get("json_validity_rate")
        schema     = ckpt_data.get("json_schema_compliance_rate")
        exact      = ckpt_data.get("json_exact_match_rate")

        row = (
            f"{label:<30} {fmt(alpaca_win, pct=True):>12} {fmt(rougeL):>8} "
            f"{fmt(bert_f1):>10} {fmt(json_val, pct=True):>12} "
            f"{fmt(schema, pct=True):>8} {fmt(exact, pct=True):>12}"
        )
        lines.append(row)

    lines.append("=" * 100)
    return "\n".join(lines)


def forgetting_analysis(summary: dict) -> dict:
    c1 = summary.get("checkpoint_1", {})
    c2 = summary.get("checkpoint_2", {})

    def delta(key):
        v1 = c1.get(key)
        v2 = c2.get(key)
        if v1 is not None and v2 is not None:
            return round(v2 - v1, 4)
        return None

    return {
        "alpaca_win_rate_delta":       delta("alpaca_win_rate_vs_baseline"),
        "alpaca_rougeL_delta":         delta("alpaca_rougeL"),
        "alpaca_bertscore_f1_delta":   delta("alpaca_bertscore_f1"),
        "json_validity_delta":         delta("json_validity_rate"),
        "json_schema_compliance_delta":delta("json_schema_compliance_rate"),
        "json_exact_match_delta":      delta("json_exact_match_rate"),
        "interpretation": (
            "Positive delta = improvement after Stage 2. "
            "Negative alpaca delta = catastrophic forgetting of Stage 1 gains."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_dir = Path("logs")

    summary = {}

    for ckpt in [0, 1, 2]:
        ckpt_data = {}

        # Alpaca auto-metrics
        alpaca_m = load_json_safe(log_dir / f"alpaca_metrics_ckpt{ckpt}.json")
        ckpt_data["alpaca_rouge1"]        = alpaca_m.get("rouge1")
        ckpt_data["alpaca_rouge2"]        = alpaca_m.get("rouge2")
        ckpt_data["alpaca_rougeL"]        = alpaca_m.get("rougeL")
        ckpt_data["alpaca_bertscore_f1"]  = alpaca_m.get("bertscore_f1")
        ckpt_data["avg_response_length"]  = alpaca_m.get("avg_response_length")

        # JSON auto-metrics
        json_m = load_json_safe(log_dir / f"json_metrics_ckpt{ckpt}.json")
        ckpt_data["json_validity_rate"]          = json_m.get("json_validity_rate")
        ckpt_data["json_schema_compliance_rate"] = json_m.get("schema_compliance_rate")
        ckpt_data["json_exact_match_rate"]       = json_m.get("exact_match_rate")
        ckpt_data["json_field_level_f1"]         = json_m.get("field_level_f1")
        ckpt_data["json_rougeL"]                 = json_m.get("rougeL")
        ckpt_data["json_task_breakdown"]         = json_m.get("task_breakdown", {})
        ckpt_data["json_error_taxonomy"]         = json_m.get("error_taxonomy", {})

        # Alpaca judge win rate vs checkpoint 0 (baseline)
        # Checkpoint 0 doesn't have a win rate vs itself; we fill in 0.5 as neutral
        if ckpt == 0:
            ckpt_data["alpaca_win_rate_vs_baseline"] = 0.5
        else:
            judge_summary = load_json_safe(log_dir / f"alpaca_judge_ckpt0_vs_ckpt{ckpt}_summary.json")
            ckpt_data["alpaca_win_rate_vs_baseline"] = judge_summary.get(f"win_rate_ckpt{ckpt}")

        # JSON judge qualitative scores
        json_judge = load_json_safe(log_dir / f"json_judge_ckpt{ckpt}_summary.json")
        ckpt_data["json_judge_avg_scores"] = json_judge.get("avg_scores", {})

        summary[f"checkpoint_{ckpt}"] = ckpt_data

    # Pairwise alpaca judge: 1 vs 2 (key forgetting comparison)
    pairwise_1v2 = load_json_safe(log_dir / "alpaca_judge_ckpt1_vs_ckpt2_summary.json")
    summary["alpaca_judge_ckpt1_vs_ckpt2"] = pairwise_1v2

    # Forgetting analysis
    forgetting = forgetting_analysis(summary)
    summary["forgetting_analysis"] = forgetting

    # Save
    with open(log_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(log_dir / "forgetting_analysis.json", "w") as f:
        json.dump(forgetting, f, indent=2)

    # Print table
    print(build_table(summary))

    print("\nFORGETTING ANALYSIS (Checkpoint 1 → Checkpoint 2):")
    print("-" * 60)
    for k, v in forgetting.items():
        if k != "interpretation":
            sign = "+" if (v or 0) >= 0 else ""
            print(f"  {k:<40}: {sign}{fmt(v)}")
    print(f"\n  Note: {forgetting['interpretation']}")

    print(f"\nFull summary → {log_dir / 'final_summary.json'}")

    # Print task breakdown for checkpoint 2
    ckpt2_task = summary.get("checkpoint_2", {}).get("json_task_breakdown", {})
    if ckpt2_task:
        print("\nJSON Task Breakdown (Checkpoint 2):")
        print(f"  {'Task':<30} {'Validity':>10} {'ExactMatch':>12}")
        print("  " + "-" * 54)
        for task, vals in ckpt2_task.items():
            print(f"  {task:<30} {vals.get('validity_rate', 0):>10.1%} "
                  f"{vals.get('exact_rate', 0):>12.1%}")


if __name__ == "__main__":
    main()
