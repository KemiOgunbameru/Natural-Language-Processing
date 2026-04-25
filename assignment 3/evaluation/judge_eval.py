"""
evaluation/judge_eval.py
========================
LLM-as-a-Judge evaluation system.

For Alpaca tasks: pairwise comparison between two checkpoints.
For JSON tasks:   qualitative scoring of individual responses.

Supports both API-based (Together AI / OpenAI-compatible) and local judge models.

Usage:
    # Pairwise: checkpoint 0 vs 1
    python evaluation/judge_eval.py --config config.yaml --mode alpaca --ckpt-a 0 --ckpt-b 1

    # Pairwise: checkpoint 1 vs 2
    python evaluation/judge_eval.py --config config.yaml --mode alpaca --ckpt-a 1 --ckpt-b 2

    # JSON qualitative scoring for a single checkpoint
    python evaluation/judge_eval.py --config config.yaml --mode json --ckpt 2
"""

import argparse
import json
import os
import random
import re
import time
import yaml
from pathlib import Path
from typing import Optional

import openai


# ---------------------------------------------------------------------------
# Config & Data helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_prompt_template(name: str, cfg: dict) -> str:
    p = Path(cfg["paths"]["prompts_dir"]) / "judge" / f"{name}.txt"
    return p.read_text().strip()


# ---------------------------------------------------------------------------
# Judge API call
# ---------------------------------------------------------------------------

def call_judge(prompt: str, cfg: dict, max_retries: int = 3) -> str:
    client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY", ""),
        base_url=cfg.get("teacher_api_base", "https://api.together.xyz/v1"),
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg["judge_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg["evaluation"]["judge_temperature"],
                max_tokens=cfg["evaluation"]["judge_max_tokens"],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [judge retry {attempt+1}] {e}")
            time.sleep(2 ** attempt)

    return ""


# ---------------------------------------------------------------------------
# Parse judge JSON output
# ---------------------------------------------------------------------------

def parse_judge_json(text: str) -> Optional[dict]:
    """Extract and parse JSON from the judge's response."""
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Alpaca pairwise evaluation
# ---------------------------------------------------------------------------

def run_alpaca_pairwise(cfg: dict, ckpt_a: int, ckpt_b: int):
    """
    For each prompt, compare the two checkpoint responses using the judge.
    Optionally swap A/B order to reduce position bias.
    """
    template = load_prompt_template("alpaca_judge", cfg)
    paths_cfg = cfg["paths"]
    eval_cfg  = cfg["evaluation"]

    responses_a = {
        ex["prompt_id"]: ex
        for ex in load_jsonl(
            str(Path(paths_cfg[f"checkpoint{ckpt_a}_dir"]) / "alpaca_responses.jsonl")
        )
    }
    responses_b = {
        ex["prompt_id"]: ex
        for ex in load_jsonl(
            str(Path(paths_cfg[f"checkpoint{ckpt_b}_dir"]) / "alpaca_responses.jsonl")
        )
    }

    prompt_ids = list(responses_a.keys())
    random.shuffle(prompt_ids)
    prompt_ids = prompt_ids[:eval_cfg["alpaca_eval_prompts"]]

    results = []
    wins_a = wins_b = ties = invalid = 0
    dim_scores_a = {d: [] for d in ["instruction_following", "correctness", "clarity",
                                     "completeness", "hallucination_risk"]}
    dim_scores_b = {d: [] for d in dim_scores_a}

    for i, pid in enumerate(prompt_ids):
        if pid not in responses_b:
            print(f"  [skip] {pid} not in checkpoint {ckpt_b} responses")
            continue

        ex_a = responses_a[pid]
        ex_b = responses_b[pid]

        # Possibly swap to mitigate position bias
        swapped = eval_cfg["judge_swap_order"] and random.random() < 0.5
        resp_1 = ex_b["response"] if swapped else ex_a["response"]
        resp_2 = ex_a["response"] if swapped else ex_b["response"]
        label_1 = f"checkpoint_{ckpt_b}" if swapped else f"checkpoint_{ckpt_a}"
        label_2 = f"checkpoint_{ckpt_a}" if swapped else f"checkpoint_{ckpt_b}"

        filled_prompt = template.format(
            instruction=ex_a["instruction"],
            input=ex_a.get("input", ""),
            response_a=resp_1,
            response_b=resp_2,
            checkpoint_a=label_1,
            checkpoint_b=label_2,
        )

        raw = call_judge(filled_prompt, cfg)
        parsed = parse_judge_json(raw)

        if parsed is None:
            print(f"  [warn] Could not parse judge output for {pid}")
            invalid += 1
            continue

        # Correct for swap
        winner_raw = parsed.get("winner", "tie").upper()
        if swapped:
            if winner_raw == "A":
                winner_raw = "B"
            elif winner_raw == "B":
                winner_raw = "A"

        if winner_raw == "A":
            wins_a += 1
        elif winner_raw == "B":
            wins_b += 1
        else:
            ties += 1

        # Collect dimension scores (correct for swap)
        scores_raw_a = parsed.get("response_a_scores", {})
        scores_raw_b = parsed.get("response_b_scores", {})
        scores_a = scores_raw_b if swapped else scores_raw_a
        scores_b = scores_raw_a if swapped else scores_raw_b

        for dim in dim_scores_a:
            if dim in scores_a:
                dim_scores_a[dim].append(scores_a[dim])
            if dim in scores_b:
                dim_scores_b[dim].append(scores_b[dim])

        results.append({
            "prompt_id":   pid,
            "checkpoint_a": f"checkpoint_{ckpt_a}",
            "checkpoint_b": f"checkpoint_{ckpt_b}",
            "winner":       winner_raw,
            "swapped":      swapped,
            "scores_a":     scores_a,
            "scores_b":     scores_b,
            "justification": parsed.get("justification", ""),
            "raw_output":   raw,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompt_ids)}] Ckpt{ckpt_a} wins: {wins_a} | "
                  f"Ckpt{ckpt_b} wins: {wins_b} | Ties: {ties}")

    n_valid = len(results)
    summary = {
        "comparison":     f"checkpoint_{ckpt_a}_vs_checkpoint_{ckpt_b}",
        "n_evaluated":    n_valid,
        "n_invalid":      invalid,
        f"wins_ckpt{ckpt_a}":   wins_a,
        f"wins_ckpt{ckpt_b}":   wins_b,
        "ties":           ties,
        f"win_rate_ckpt{ckpt_a}": round(wins_a / n_valid, 4) if n_valid else 0,
        f"win_rate_ckpt{ckpt_b}": round(wins_b / n_valid, 4) if n_valid else 0,
        "tie_rate":       round(ties / n_valid, 4) if n_valid else 0,
        f"avg_scores_ckpt{ckpt_a}": {
            d: round(sum(v)/len(v), 2) for d, v in dim_scores_a.items() if v
        },
        f"avg_scores_ckpt{ckpt_b}": {
            d: round(sum(v)/len(v), 2) for d, v in dim_scores_b.items() if v
        },
    }

    # Save
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    tag = f"alpaca_judge_ckpt{ckpt_a}_vs_ckpt{ckpt_b}"

    with open(out_dir / f"{tag}_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Alpaca Judge: Checkpoint {ckpt_a} vs Checkpoint {ckpt_b}")
    print(f"  Ckpt {ckpt_a} win rate: {summary[f'win_rate_ckpt{ckpt_a}']:.1%}")
    print(f"  Ckpt {ckpt_b} win rate: {summary[f'win_rate_ckpt{ckpt_b}']:.1%}")
    print(f"  Tie rate:      {summary['tie_rate']:.1%}")
    print(f"  Results → {out_dir / f'{tag}_results.jsonl'}")

    return summary


# ---------------------------------------------------------------------------
# JSON qualitative scoring
# ---------------------------------------------------------------------------

def run_json_qualitative(cfg: dict, ckpt: int):
    template = load_prompt_template("json_judge", cfg)
    paths_cfg = cfg["paths"]
    eval_cfg  = cfg["evaluation"]

    responses = load_jsonl(
        str(Path(paths_cfg[f"checkpoint{ckpt}_dir"]) / "json_responses.jsonl")
    )

    results = []
    all_scores = {d: [] for d in ["instruction_following", "correctness", "clarity",
                                   "completeness", "structured_output_validity",
                                   "hallucination_risk"]}

    for i, ex in enumerate(responses[:eval_cfg["json_eval_prompts"]]):
        filled_prompt = template.format(
            instruction=ex["instruction"],
            input=ex.get("input", ""),
            response=ex["response"],
            task_type=ex.get("task_type", "json"),
        )

        raw    = call_judge(filled_prompt, cfg)
        parsed = parse_judge_json(raw)

        if parsed is None:
            print(f"  [warn] Could not parse judge output for {ex.get('prompt_id', i)}")
            continue

        scores = parsed.get("scores", {})
        for dim in all_scores:
            if dim in scores:
                all_scores[dim].append(scores[dim])

        results.append({
            "prompt_id":  ex.get("prompt_id", f"json_eval_{i:04d}"),
            "task_type":  ex.get("task_type", "json"),
            "checkpoint": f"checkpoint_{ckpt}",
            "scores":     scores,
            "verdict":    parsed.get("verdict", ""),
            "justification": parsed.get("justification", ""),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{min(len(responses), eval_cfg['json_eval_prompts'])}] scored")

    summary = {
        "checkpoint":    f"checkpoint_{ckpt}",
        "n_evaluated":   len(results),
        "avg_scores":    {d: round(sum(v)/len(v), 2) for d, v in all_scores.items() if v},
    }

    out_dir = Path("logs")
    tag     = f"json_judge_ckpt{ckpt}"
    with open(out_dir / f"{tag}_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"JSON Judge: Checkpoint {ckpt}")
    for d, v in summary["avg_scores"].items():
        print(f"  {d}: {v}/5")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode",   choices=["alpaca", "json"], required=True)
    parser.add_argument("--ckpt-a", type=int, default=0)
    parser.add_argument("--ckpt-b", type=int, default=1)
    parser.add_argument("--ckpt",   type=int, default=2,
                        help="For --mode json: which checkpoint to score")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = load_config(args.config)

    if args.mode == "alpaca":
        run_alpaca_pairwise(cfg, args.ckpt_a, args.ckpt_b)
    else:
        run_json_qualitative(cfg, args.ckpt)


if __name__ == "__main__":
    main()
