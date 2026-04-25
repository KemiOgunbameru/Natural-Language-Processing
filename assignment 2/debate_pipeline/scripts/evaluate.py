"""
evaluate.py — Read saved logs and produce all tables and figures.

Usage:
    python scripts/evaluate.py --results_dir results/
"""

import argparse
import json
from pathlib import Path


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_results_table(summary: dict, domain: str) -> None:
    print(f"\n{'='*55}")
    print(f"  Results: {domain.replace('_', ' ').title()}")
    print(f"{'='*55}")
    print(f"  {'Method':<22} {'Accuracy':>10}  {'N':>6}")
    print(f"  {'-'*40}")
    print(
        f"  {'Debate Pipeline':<22} "
        f"{summary['debate']['accuracy']:>10.1%}  "
        f"{summary['debate']['n']:>6}"
    )
    print(
        f"  {'Direct QA (CoT)':<22} "
        f"{summary['direct_qa']['accuracy']:>10.1%}  "
        f"{summary['direct_qa']['n']:>6}"
    )
    print(
        f"  {'Self-Consistency':<22} "
        f"{summary['self_consistency']['accuracy']:>10.1%}  "
        f"{summary['self_consistency']['n']:>6}"
    )
    print(f"\n  Debate avg. rounds: {summary['debate']['avg_rounds']:.2f}")
    print(f"\n  Confidence vs. Accuracy (Debate):")
    print(f"  {'Confidence':<15} {'Accuracy':>10}")
    print(f"  {'-'*26}")
    for conf, acc in sorted(summary["debate"]["confidence_accuracy"].items()):
        if acc is not None:
            print(f"  {'Level ' + str(conf):<15} {acc:>10.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summaries = list(results_dir.glob("summary_*.json"))

    if not summaries:
        print(f"No summary files found in {results_dir}/")
        return

    for path in summaries:
        domain = path.stem.replace("summary_", "")
        summary = load_summary(str(path))
        print_results_table(summary, domain)


if __name__ == "__main__":
    main()
