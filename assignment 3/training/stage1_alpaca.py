"""
training/stage1_alpaca.py
=========================
Stage 1: QLoRA fine-tuning of Phi-3.5 Mini on Alpaca instruction data.

Saves the LoRA adapter checkpoint to outputs/checkpoint1/.

Usage:
    python training/stage1_alpaca.py --config config.yaml

On UTSA HPC, this script is launched via hpc/stage1_train.slurm.
"""

import argparse
import json
import logging
import os
import yaml
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def make_dataset(data_path: str, tokenizer, max_seq_length: int) -> Dataset:
    """Load JSONL and convert to HuggingFace Dataset."""
    raw = load_jsonl(data_path)

    # Handle both raw schema (instruction/input/output) and pre-formatted (text)
    if "text" in raw[0]:
        texts = [ex["text"] for ex in raw]
    else:
        texts = [format_prompt(ex) for ex in raw]

    dataset = Dataset.from_dict({"text": texts})
    logger.info(f"Loaded {len(dataset)} examples from {data_path}")
    return dataset


def format_prompt(ex: dict) -> str:
    system = "You are a helpful assistant."
    user_turn = ex["instruction"]
    if ex.get("input"):
        user_turn += f"\n\n{ex['input']}"
    return (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user_turn}<|end|>\n"
        f"<|assistant|>\n{ex['output']}<|end|>"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, lora_cfg: dict, precision: str):
    logger.info(f"Loading base model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Prevents infinite recursion in generation

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        inference_mode=False,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.entries  = []
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {"step": state.global_step, **logs}
            self.entries.append(entry)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    stage1_cfg = cfg["training"]["stage1"]
    train_cfg  = cfg["training"]
    lora_cfg   = cfg["lora"]
    paths_cfg  = cfg["paths"]

    output_dir = Path(stage1_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(paths_cfg["logs_dir"]) / "stage1_loss.jsonl"

    model, tokenizer = load_model_and_tokenizer(
        cfg["student_model"], lora_cfg, train_cfg["precision"]
    )

    train_dataset = make_dataset(
        stage1_cfg["dataset_path"], tokenizer, train_cfg["max_seq_length"]
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=stage1_cfg["epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=stage1_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=False,
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        report_to="none",
        run_name="stage1_alpaca",
        optim="paged_adamw_32bit",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=train_cfg["max_seq_length"],
        callbacks=[LossLoggerCallback(log_path)],
    )

    logger.info("=" * 60)
    logger.info("Stage 1: Alpaca Fine-Tuning")
    logger.info(f"  Model:      {cfg['student_model']}")
    logger.info(f"  Dataset:    {stage1_cfg['dataset_path']}")
    logger.info(f"  Epochs:     {stage1_cfg['epochs']}")
    logger.info(f"  LR:         {stage1_cfg['learning_rate']}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info("=" * 60)

    trainer.train()

    # Save final adapter
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Stage 1 adapter saved to {output_dir}")

    # Save training metadata
    meta = {
        "stage":          "stage1_alpaca",
        "model":          cfg["student_model"],
        "epochs":         stage1_cfg["epochs"],
        "learning_rate":  stage1_cfg["learning_rate"],
        "lora_r":         lora_cfg["r"],
        "lora_alpha":     lora_cfg["alpha"],
        "dataset":        stage1_cfg["dataset_path"],
        "final_loss":     trainer.state.log_history[-1].get("loss", None),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Stage 1 complete.")


if __name__ == "__main__":
    main()
