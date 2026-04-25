"""
inference/generate_outputs.py
==============================
Generates responses for the Alpaca and JSON eval sets at all three
checkpoints (0 = untuned, 1 = after Stage 1, 2 = after Stage 2).

Outputs are saved to outputs/checkpoint{N}/ for use by the judge and
metrics evaluators.

Usage:
    # All checkpoints
    python inference/generate_outputs.py --config config.yaml --checkpoint all

    # Single checkpoint
    python inference/generate_outputs.py --config config.yaml --checkpoint 1
"""

import argparse
import json
import logging
import yaml
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def format_prompt_no_output(ex: dict, system: str) -> str:
    """Build the model input (without the expected output)."""
    user_turn = ex["instruction"]
    if ex.get("input"):
        user_turn += f"\n\n{ex['input']}"
    return (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user_turn}<|end|>\n"
        f"<|assistant|>\n"
    )


def load_model(
    base_model_name: str,
    adapter_path: Optional[str] = None,
) -> tuple:
    """Load quantised base model and optional LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # Left-pad for generation

    if adapter_path:
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_responses(
    model,
    tokenizer,
    examples: list,
    system: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    batch_size: int = 4,
) -> list:
    results = []

    for i in tqdm(range(0, len(examples), batch_size), desc="Generating"):
        batch = examples[i : i + batch_size]
        prompts = [format_prompt_no_output(ex, system) for ex in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048 - max_new_tokens,
        ).to(model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        outputs = model.generate(**inputs, **gen_kwargs)

        for j, (output_ids, prompt_ids) in enumerate(
            zip(outputs, inputs["input_ids"])
        ):
            # Decode only the newly generated tokens
            new_ids = output_ids[len(prompt_ids):]
            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            ex = batch[j]
            results.append(
                {
                    "prompt_id":   ex.get("prompt_id", f"{i+j}"),
                    "instruction": ex["instruction"],
                    "input":       ex.get("input", ""),
                    "reference":   ex.get("output", ""),
                    "response":    response,
                    "task_type":   ex.get("task_type", "alpaca"),
                }
            )

    return results


def run_checkpoint(
    checkpoint_id: int,
    cfg: dict,
    alpaca_eval: list,
    json_eval: list,
):
    paths = cfg["paths"]
    eval_cfg = cfg["evaluation"]
    base_name = cfg["student_model"]

    adapter_map = {
        0: None,
        1: paths["checkpoint1_dir"],
        2: paths["checkpoint2_dir"],
    }

    adapter_path = adapter_map[checkpoint_id]
    out_dir = Path(paths[f"checkpoint{checkpoint_id}_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Checkpoint {checkpoint_id}: {adapter_path or 'untuned base'}")
    logger.info("="*60)

    model, tokenizer = load_model(base_name, adapter_path)

    # Alpaca responses
    alpaca_system = "You are a helpful assistant."
    alpaca_results = generate_responses(
        model, tokenizer, alpaca_eval,
        system=alpaca_system,
        max_new_tokens=eval_cfg["inference_max_new_tokens"],
        temperature=eval_cfg["inference_temperature"],
        do_sample=eval_cfg["inference_do_sample"],
    )
    alpaca_out = out_dir / "alpaca_responses.jsonl"
    with open(alpaca_out, "w") as f:
        for r in alpaca_results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved {len(alpaca_results)} Alpaca responses → {alpaca_out}")

    # JSON responses
    json_system = "You are a helpful assistant. Always respond with valid JSON when asked."
    json_results = generate_responses(
        model, tokenizer, json_eval,
        system=json_system,
        max_new_tokens=eval_cfg["inference_max_new_tokens"],
        temperature=eval_cfg["inference_temperature"],
        do_sample=eval_cfg["inference_do_sample"],
    )
    json_out = out_dir / "json_responses.jsonl"
    with open(json_out, "w") as f:
        for r in json_results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved {len(json_results)} JSON responses → {json_out}")

    # Free GPU memory before next checkpoint
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default="all",
                        help="0, 1, 2, or 'all'")
    args = parser.parse_args()

    cfg = load_config(args.config)

    alpaca_eval = load_jsonl("data/alpaca_eval.jsonl")
    json_eval   = load_jsonl("data/json_instruct_eval.jsonl")

    # Add stable prompt IDs
    for i, ex in enumerate(alpaca_eval):
        ex["prompt_id"] = f"alpaca_eval_{i:04d}"
    for i, ex in enumerate(json_eval):
        ex["prompt_id"] = f"json_eval_{i:04d}"

    checkpoints = (
        [0, 1, 2] if args.checkpoint == "all"
        else [int(args.checkpoint)]
    )

    for ckpt in checkpoints:
        run_checkpoint(ckpt, cfg, alpaca_eval, json_eval)

    logger.info("\nAll inference complete.")


if __name__ == "__main__":
    main()
