"""
run_debate.py — Run a single debate interactively or with CLI args.

Usage:
    python scripts/run_debate.py \
        --question "Did Shakespeare live during the same century as Galileo?" \
        --answer "yes" \
        --position_a "yes" \
        --position_b "no" \
        --context "Shakespeare: 1564-1616. Galileo: 1564-1642."
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import DebateOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Run a single LLM debate")
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", required=True, help="Ground truth answer")
    parser.add_argument("--position_a", required=True)
    parser.add_argument("--position_b", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--id", default="manual_run")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY environment variable first.")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    orchestrator = DebateOrchestrator(config)
    result = orchestrator.run_debate(
        question=args.question,
        ground_truth=args.answer,
        position_a=args.position_a,
        position_b=args.position_b,
        context=args.context,
        question_id=args.id,
    )

    print("\n" + "="*60)
    print("DEBATE COMPLETE")
    print("="*60)
    print(f"Verdict:       {result['phases']['phase3']['verdict']}")
    print(f"Ground Truth:  {result['phases']['phase4']['ground_truth']}")
    print(f"Correct:       {'✓ YES' if result['correct'] else '✗ NO'}")
    print(f"Confidence:    {result['judge_confidence']}/5")
    print(f"Rounds:        {result['num_rounds_completed']}")
    print(f"\nJudge Reasoning:")
    print(result["phases"]["phase3"].get("reasoning_summary", "N/A"))


if __name__ == "__main__":
    main()
