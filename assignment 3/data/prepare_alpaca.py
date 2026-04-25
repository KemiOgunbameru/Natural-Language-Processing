"""
data/prepare_alpaca.py
======================
Downloads the Alpaca instruction dataset, cleans it, normalises it into
the shared (instruction, input, output) schema, and splits it into a
training set and a held-out evaluation set.

Usage:
    python data/prepare_alpaca.py --config config.yaml
"""

import argparse
import json
import random
import re
import yaml
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """Strip common artefacts from alpaca source data."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Remove leading/trailing quotation marks sometimes present
    text = re.sub(r'^["\']|["\']$', "", text)
    return text.strip()


def is_valid_example(ex: dict) -> bool:
    """Filter out malformed or trivially empty examples."""
    instruction = clean_text(ex.get("instruction", ""))
    output = clean_text(ex.get("output", ""))
    if len(instruction) < 10:
        return False
    if len(output) < 5:
        return False
    # Drop examples where the output is just "N/A" or similar
    if output.lower() in {"n/a", "none", "na", "null", ""}:
        return False
    return True


def normalise(ex: dict) -> dict:
    """Return a clean (instruction, input, output) dict."""
    return {
        "instruction": clean_text(ex.get("instruction", "")),
        "input":       clean_text(ex.get("input", "")),
        "output":      clean_text(ex.get("output", "")),
    }


def format_prompt(ex: dict) -> str:
    """Convert normalised example to Phi-3.5 chat format."""
    system = "You are a helpful assistant."
    user_turn = ex["instruction"]
    if ex["input"]:
        user_turn += f"\n\n{ex['input']}"
    return (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user_turn}<|end|>\n"
        f"<|assistant|>\n{ex['output']}<|end|>"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(args.seed)

    data_cfg   = cfg["data"]
    out_dir    = Path(cfg["paths"]["outputs_dir"])
    data_dir   = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_size = data_cfg["alpaca_train_size"]
    eval_size  = data_cfg["alpaca_eval_size"]

    print(f"[prepare_alpaca] Loading dataset: {data_cfg['alpaca_source']}")
    ds = load_dataset(data_cfg["alpaca_source"], split="train")

    # Filter and normalise
    examples = [normalise(ex) for ex in ds if is_valid_example(ex)]
    print(f"[prepare_alpaca] Valid examples after filtering: {len(examples)}")

    random.shuffle(examples)

    need = train_size + eval_size
    if len(examples) < need:
        raise ValueError(
            f"Not enough valid examples: have {len(examples)}, need {need}"
        )

    eval_examples  = examples[:eval_size]
    train_examples = examples[eval_size : eval_size + train_size]

    def save_jsonl(path: Path, items: list):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"[prepare_alpaca] Saved {len(items)} examples → {path}")

    # Raw schema files (used by training scripts)
    save_jsonl(data_dir / "alpaca_train.jsonl", train_examples)
    save_jsonl(data_dir / "alpaca_eval.jsonl",  eval_examples)

    # Also save prompt-formatted versions for quick inspection
    formatted_train = [{"text": format_prompt(ex)} for ex in train_examples]
    formatted_eval  = [{"text": format_prompt(ex)} for ex in eval_examples]
    save_jsonl(data_dir / "alpaca_train_formatted.jsonl", formatted_train)
    save_jsonl(data_dir / "alpaca_eval_formatted.jsonl",  formatted_eval)

    print("[prepare_alpaca] Done.")


if __name__ == "__main__":
    main()
