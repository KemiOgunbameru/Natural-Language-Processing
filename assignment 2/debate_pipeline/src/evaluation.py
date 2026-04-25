"""
evaluation.py — Baselines and metrics.

Implements:
  1. Direct QA baseline (single model, CoT, zero-shot)
  2. Self-Consistency baseline (majority vote over N samples)
  3. Accuracy, confidence calibration, and per-difficulty metrics
"""

from __future__ import annotations
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from openai import OpenAI


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates (inline for baselines)
# ──────────────────────────────────────────────────────────────────────────────

DIRECT_QA_PROMPT = """\
Answer the following question using step-by-step Chain-of-Thought reasoning.

Question: {question}

Context: {context}

Possible answers: {answers}

Instructions:
1. Think through the problem carefully before answering.
2. Consider relevant facts, definitions, and logical implications.
3. State your final answer on the LAST line in the exact format: ANSWER: [your answer]

Your response:
"""

SELF_CONSISTENCY_PROMPT = """\
Answer the following question. Think step-by-step, then state your final answer.

Question: {question}

Context: {context}

Possible answers: {answers}

Provide your answer in the format: ANSWER: [your answer]
"""


# ──────────────────────────────────────────────────────────────────────────────
# Baselines
# ──────────────────────────────────────────────────────────────────────────────

class BaselineEvaluator:
    """Runs Direct QA and Self-Consistency baselines."""

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.client = OpenAI(api_key=config["api"]["key"], base_url=config["api"]["base_url"]), base_url="http://149.165.173.247:8888/v1")
        self.model = config["model"]["baseline"]
        self.temperature = config["generation"]["baseline_temperature"]
        self.max_tokens = config["generation"]["baseline_max_tokens"]
        self.sc_samples = config["evaluation"]["self_consistency_samples"]
        self.log_dir = Path(config["logging"]["log_dir"])

    # ── Direct QA ─────────────────────────────────────────────────────────────

    def direct_qa(
        self,
        question: str,
        ground_truth: str,
        answers: list[str],
        context: str = "",
        question_id: Optional[str] = None,
    ) -> dict:
        """Single-model CoT answer."""
        prompt = DIRECT_QA_PROMPT.format(
            question=question,
            context=context or "No additional context.",
            answers=" / ".join(answers),
        )
        raw = self._call_api(prompt, temperature=self.temperature)
        answer = self._extract_answer(raw, answers)
        correct = self._is_correct(answer, ground_truth)

        result = {
            "method": "direct_qa",
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": answer,
            "correct": correct,
            "raw_response": raw,
        }
        self._maybe_save(result, f"direct_{question_id}")
        return result

    # ── Self-Consistency ──────────────────────────────────────────────────────

    def self_consistency(
        self,
        question: str,
        ground_truth: str,
        answers: list[str],
        context: str = "",
        question_id: Optional[str] = None,
    ) -> dict:
        """Sample N answers, take majority vote."""
        prompt = SELF_CONSISTENCY_PROMPT.format(
            question=question,
            context=context or "No additional context.",
            answers=" / ".join(answers),
        )
        samples = []
        for _ in range(self.sc_samples):
            raw = self._call_api(prompt, temperature=self.temperature)
            ans = self._extract_answer(raw, answers)
            samples.append(ans)

        # Majority vote
        vote_counts = Counter(samples)
        majority_answer = vote_counts.most_common(1)[0][0]
        correct = self._is_correct(majority_answer, ground_truth)

        result = {
            "method": "self_consistency",
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": majority_answer,
            "correct": correct,
            "all_samples": samples,
            "vote_distribution": dict(vote_counts),
            "num_samples": self.sc_samples,
        }
        self._maybe_save(result, f"sc_{question_id}")
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _call_api(self, prompt: str, temperature: float) -> str:
    for attempt in range(3):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt * 3)
    raise RuntimeError("Baseline API call failed")

    @staticmethod
    def _extract_answer(raw: str, valid_answers: list[str]) -> str:
        """Extract the answer after 'ANSWER:' token, or best fuzzy match."""
        match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            # Return exact match if found
            for ans in valid_answers:
                if ans.lower() in candidate.lower() or candidate.lower() in ans.lower():
                    return ans
            return candidate

        # Fallback: find any valid answer mentioned last
        last_match = None
        for ans in valid_answers:
            if ans.lower() in raw.lower():
                last_match = ans
        return last_match or valid_answers[0]

    @staticmethod
    def _is_correct(predicted: str, ground_truth: str) -> bool:
        p = predicted.strip().lower()
        g = ground_truth.strip().lower()
        return (p == g) or (g in p) or (p in g)

    def _maybe_save(self, result: dict, name: str) -> None:
        if self.cfg["logging"]["save_transcripts"]:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"{name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_accuracy(results: list[dict]) -> float:
    """Fraction of correct answers."""
    if not results:
        return 0.0
    return sum(r["correct"] for r in results) / len(results)


def compute_confidence_accuracy(results: list[dict]) -> dict:
    """
    Break down debate accuracy by judge confidence score (1–5).
    Returns {confidence_level: accuracy} dict.
    """
    buckets: dict[int, list[bool]] = {i: [] for i in range(1, 6)}
    for r in results:
        conf = r.get("judge_confidence", -1)
        if isinstance(conf, int) and 1 <= conf <= 5:
            buckets[conf].append(r["correct"])
    return {
        k: (sum(v) / len(v) if v else None)
        for k, v in buckets.items()
    }


def aggregate_results(
    debate_results: list[dict],
    direct_qa_results: list[dict],
    sc_results: list[dict],
) -> dict:
    """Compute and return all summary metrics."""
    return {
        "debate": {
            "accuracy": compute_accuracy(debate_results),
            "n": len(debate_results),
            "confidence_accuracy": compute_confidence_accuracy(debate_results),
            "avg_rounds": (
                sum(r["num_rounds_completed"] for r in debate_results)
                / max(len(debate_results), 1)
            ),
        },
        "direct_qa": {
            "accuracy": compute_accuracy(direct_qa_results),
            "n": len(direct_qa_results),
        },
        "self_consistency": {
            "accuracy": compute_accuracy(sc_results),
            "n": len(sc_results),
        },
    }


def save_summary(summary: dict, path: str = "results/summary.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {path}")
