"""
run_experiments.py — Run all required experiments.

Usage:
    python scripts/run_experiments.py --domain commonsense_qa --n 20
    python scripts/run_experiments.py --domain fact_verification --n 100
    python scripts/run_experiments.py --all
"""

import argparse
import json
import sys
import os
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import DebateOrchestrator
from src.evaluation import BaselineEvaluator, aggregate_results, save_summary


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_questions(domain: str, n: int, data_dir: str = "data") -> list[dict]:
    path = Path(data_dir) / "sample_questions.json"
    with open(path) as f:
        all_q = json.load(f)

    filtered = [q for q in all_q if q["domain"] == domain]
    # Repeat to fill n if needed (for testing with small datasets)
    while len(filtered) < n:
        filtered = filtered * 2
    return filtered[:n]


def run_all_experiments(config: dict, domain: str, n: int) -> dict:
    questions = load_questions(domain, n, config["paths"]["data_dir"])
    print(f"\nRunning experiments on {len(questions)} {domain} questions")
    print("="*60)

    orchestrator = DebateOrchestrator(config)
    evaluator = BaselineEvaluator(config)

    debate_results = []
    direct_qa_results = []
    sc_results = []

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['id']}")

        # 1. Debate pipeline
        debate_res = orchestrator.run_debate(
            question=q["question"],
            ground_truth=q["answer"],
            position_a=q["position_a"],
            position_b=q["position_b"],
            context=q.get("context", ""),
            question_id=q["id"],
        )
        debate_results.append(debate_res)

        # 2. Direct QA baseline
        direct_res = evaluator.direct_qa(
            question=q["question"],
            ground_truth=q["answer"],
            answers=[q["position_a"], q["position_b"]],
            context=q.get("context", ""),
            question_id=q["id"],
        )
        direct_qa_results.append(direct_res)

        # 3. Self-Consistency baseline
        sc_res = evaluator.self_consistency(
            question=q["question"],
            ground_truth=q["answer"],
            answers=[q["position_a"], q["position_b"]],
            context=q.get("context", ""),
            question_id=q["id"],
        )
        sc_results.append(sc_res)

    summary = aggregate_results(debate_results, direct_qa_results, sc_results)
    save_summary(summary, f"results/summary_{domain}.json")

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Debate Pipeline:    {summary['debate']['accuracy']:.1%}")
    print(f"Direct QA:          {summary['direct_qa']['accuracy']:.1%}")
    print(f"Self-Consistency:   {summary['self_consistency']['accuracy']:.1%}")
    print(f"\nDebate avg rounds:  {summary['debate']['avg_rounds']:.1f}")
    print(f"\nConfidence vs Accuracy:")
    for conf, acc in summary["debate"]["confidence_accuracy"].items():
        if acc is not None:
            print(f"  Confidence {conf}: {acc:.1%}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run LLM Debate experiments")
    parser.add_argument(
        "--domain",
        choices=["commonsense_qa", "fact_verification"],
        default="commonsense_qa",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of questions")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--all", action="store_true", help="Run both domains")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    config = load_config(args.config)

    if args.all:
        for domain in ["commonsense_qa", "fact_verification"]:
            run_all_experiments(config, domain, args.n)
    else:
        run_all_experiments(config, args.domain, args.n)


if __name__ == "__main__":
    main()
