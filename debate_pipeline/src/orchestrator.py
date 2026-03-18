"""
orchestrator.py — Debate Orchestrator.

Implements the full 4-phase debate protocol:
  Phase 1 — Initialization (independent initial positions)
  Phase 2 — Multi-Round Debate (N rounds, with adaptive early stopping)
  Phase 3 — Judgment
  Phase 4 — Evaluation (comparison against ground truth)
"""

from __future__ import annotations
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agents.debater import DebaterAgent
from src.agents.judge import JudgeAgent


class DebateOrchestrator:
    """
    Manages one complete debate for a single question.

    Parameters
    ----------
    config : dict
        Parsed config.yaml contents.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.num_rounds = config["debate"]["num_rounds"]
        self.early_stop = config["debate"]["early_stop_consensus"]
        self.consensus_window = config["debate"]["consensus_window"]
        self.log_dir = Path(config["logging"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_debate(
        self,
        question: str,
        ground_truth: str,
        position_a: str,
        position_b: str,
        context: str = "",
        question_id: Optional[str] = None,
    ) -> dict:
        """
        Run the complete 4-phase debate for a single question.

        Returns a result dict suitable for JSON logging and evaluation.
        """
        q_id = question_id or f"q_{int(time.time())}"
        result = {
            "question_id": q_id,
            "question": question,
            "ground_truth": ground_truth,
            "position_a": position_a,
            "position_b": position_b,
            "context": context,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "phases": {},
        }

        print(f"\n{'='*60}")
        print(f"Question: {question[:80]}...")
        print(f"Positions — A: {position_a} | B: {position_b}")
        print(f"Ground Truth: {ground_truth}")

        # ── Phase 1: Initialization ──────────────────────────────────
        result["phases"]["phase1"] = self._phase_1(
            question, context, position_a, position_b
        )

        # Check for immediate consensus
        init_a = result["phases"]["phase1"]["initial_a"]["position"]
        init_b = result["phases"]["phase1"]["initial_b"]["position"]
        if self._positions_agree(init_a, init_b):
            print("  → Consensus at initialization — skipping debate.")
            result["phases"]["phase1"]["consensus"] = True
            result["phases"]["phase2"] = {"skipped": True, "rounds": []}
        else:
            result["phases"]["phase1"]["consensus"] = False
            # ── Phase 2: Multi-Round Debate ──────────────────────────
            result["phases"]["phase2"] = self._phase_2(
                question, context, position_a, position_b
            )

        # ── Phase 3: Judgment ────────────────────────────────────────
        debate_history = result["phases"]["phase2"].get("rounds", [])
        result["phases"]["phase3"] = self._phase_3(
            question, position_a, position_b, debate_history
        )

        # ── Phase 4: Evaluation ──────────────────────────────────────
        result["phases"]["phase4"] = self._phase_4(
            verdict=result["phases"]["phase3"]["verdict"],
            ground_truth=ground_truth,
            position_a=position_a,
            position_b=position_b,
        )

        result["correct"] = result["phases"]["phase4"]["correct"]
        result["judge_confidence"] = result["phases"]["phase3"].get("confidence", -1)
        result["num_rounds_completed"] = len(debate_history)

        # Save transcript
        if self.cfg["logging"]["save_transcripts"]:
            self._save_transcript(result, q_id)

        status = "✓" if result["correct"] else "✗"
        print(
            f"  {status} Verdict: {result['phases']['phase3']['verdict']} "
            f"(confidence: {result['judge_confidence']}/5)"
        )
        return result

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _phase_1(
        self,
        question: str,
        context: str,
        position_a: str,
        position_b: str,
    ) -> dict:
        """Each debater generates an independent initial position."""
        print("  Phase 1: Initializing positions...")
        agent_a = self._make_debater("A", position_a, position_b)
        agent_b = self._make_debater("B", position_b, position_a)

        initial_a = agent_a.generate_argument(question, context, [], round_num=0)
        initial_b = agent_b.generate_argument(question, context, [], round_num=0)

        return {"initial_a": initial_a, "initial_b": initial_b}

    def _phase_2(
        self,
        question: str,
        context: str,
        position_a: str,
        position_b: str,
    ) -> dict:
        """Multi-round adversarial debate with adaptive stopping."""
        print(f"  Phase 2: Starting {self.num_rounds}-round debate...")
        agent_a = self._make_debater("A", position_a, position_b)
        agent_b = self._make_debater("B", position_b, position_a)

        history: list[dict] = []
        consecutive_consensus = 0

        for rnd in range(1, self.num_rounds + 1):
            print(f"    Round {rnd}/{self.num_rounds}")

            # Debater A argues
            turn_a = agent_a.generate_argument(question, context, history, rnd)
            history.append(turn_a)

            # Debater B responds
            turn_b = agent_b.generate_argument(question, context, history, rnd)
            history.append(turn_b)

            # Check adaptive stopping
            if self.early_stop and self._positions_agree(
                turn_a["position"], turn_b["position"]
            ):
                consecutive_consensus += 1
                if consecutive_consensus >= self.consensus_window:
                    print(
                        f"    → Early stop: consensus for {consecutive_consensus} "
                        f"consecutive rounds."
                    )
                    break
            else:
                consecutive_consensus = 0

        return {"rounds": history, "num_rounds": rnd}

    def _phase_3(
        self,
        question: str,
        position_a: str,
        position_b: str,
        debate_history: list[dict],
    ) -> dict:
        """Judge evaluates the full transcript."""
        print("  Phase 3: Judge evaluating...")
        judge = JudgeAgent(self.cfg)
        verdict = judge.judge_debate(question, position_a, position_b, debate_history)
        return verdict

    def _phase_4(
        self,
        verdict: str,
        ground_truth: str,
        position_a: str,
        position_b: str,
    ) -> dict:
        """Compare judge verdict against ground truth."""
        # Normalize for comparison
        v = verdict.strip().lower()
        gt = ground_truth.strip().lower()

        correct = (v == gt) or (gt in v) or (v in gt)
        return {
            "verdict": verdict,
            "ground_truth": ground_truth,
            "correct": correct,
            "position_a": position_a,
            "position_b": position_b,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_debater(
        self, role: str, position: str, opponent_position: str
    ) -> DebaterAgent:
        return DebaterAgent(role, position, opponent_position, self.cfg)

    @staticmethod
    def _positions_agree(pos_a: str, pos_b: str) -> bool:
        return pos_a.strip().lower() == pos_b.strip().lower()

    def _save_transcript(self, result: dict, q_id: str) -> None:
        path = self.log_dir / f"debate_{q_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
