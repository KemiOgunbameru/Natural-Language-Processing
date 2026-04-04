"""
training/stage2_json_instruct.py
================================
Stage 2: Continue QLoRA fine-tuning from the Stage 1 Alpaca checkpoint
on the teacher-generated JSON Instruct dataset.

Loads the Stage 1 LoRA adapter and continues training on JSON data,
then saves the Stage 2 adapter to outputs/checkpoint2/.

Usage:
    python training/stage2_json_instruct.py --config config.yaml

On UTSA HPC, this script is launched via hpc/stage2_train.slurm.
"""

import argparse
import json
import logging
import yaml
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def format_prompt(ex: dict) -> str:
    system = "You are a helpful assistant. Always respond with valid JSON when asked."
    user_turn = ex["instruction"]
    if ex.get("input"):
        user_turn += f"\n\n{ex['input']}"
    return (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user_turn}<|end|>\n"
        f"<|assistant|>\n{ex['output']}<|end|>"
    )


def make_dataset(data_path: str) -> Dataset:
    raw = load_jsonl(data_path)
    if "text" in raw[0]:
        texts = [ex["text"] for ex in raw]
    else:
        texts = [format_prompt(ex) for ex in raw]
    ds = Dataset.from_dict({"text": texts})
    logger.info(f"Loaded {len(ds)} examples from {data_path}")
    return ds


class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {"step": state.global_step, **logs}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    stage2_cfg = cfg["training"]["stage2"]
    train_cfg  = cfg["training"]
    lora_cfg   = cfg["lora"]
    paths_cfg  = cfg["paths"]

    output_dir      = Path(stage2_cfg["output_dir"])
    stage1_ckpt_dir = Path(stage2_cfg["load_from"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(paths_cfg["logs_dir"]) / "stage2_loss.jsonl"

    logger.info("=" * 60)
    logger.info("Stage 2: JSON Instruct Fine-Tuning")
    logger.info(f"  Base model:     {cfg['student_model']}")
    logger.info(f"  Stage 1 ckpt:   {stage1_ckpt_dir}")
    logger.info(f"  Dataset:        {stage2_cfg['dataset_path']}")
    logger.info(f"  Epochs:         {stage2_cfg['epochs']}")
    logger.info(f"  LR:             {stage2_cfg['learning_rate']}")
    logger.info(f"  Output dir:     {output_dir}")
    logger.info("=" * 60)

    # --- Load base model in 4-bit ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["student_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["student_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = prepare_model_for_kbit_training(base_model)

    # --- Load Stage 1 LoRA adapter and merge approach ---
    # We load the Stage 1 PEFT adapter and continue training with a fresh LoRA
    # on top (sequential fine-tuning: the weights from stage 1 are the starting point).
    logger.info(f"Loading Stage 1 LoRA adapter from {stage1_ckpt_dir}")
    model = PeftModel.from_pretrained(base_model, str(stage1_ckpt_dir), is_trainable=False)

    # Merge Stage 1 adapter into base weights for a clean starting point
    logger.info("Merging Stage 1 adapter into base weights...")
    model = model.merge_and_unload()

    # Prepare again after merge
    model = prepare_model_for_kbit_training(model)

    # Attach a fresh Stage 2 LoRA adapter
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

    # --- Dataset ---
    train_dataset = make_dataset(stage2_cfg["dataset_path"])

    # --- Training ---
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=stage2_cfg["epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=stage2_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=2,
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        report_to="none",
        run_name="stage2_json_instruct",
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

    trainer.train()

    # Save Stage 2 adapter
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Stage 2 adapter saved to {output_dir}")

    meta = {
        "stage":          "stage2_json_instruct",
        "model":          cfg["student_model"],
        "stage1_ckpt":    str(stage1_ckpt_dir),
        "epochs":         stage2_cfg["epochs"],
        "learning_rate":  stage2_cfg["learning_rate"],
        "lora_r":         lora_cfg["r"],
        "lora_alpha":     lora_cfg["alpha"],
        "dataset":        stage2_cfg["dataset_path"],
        "final_loss":     trainer.state.log_history[-1].get("loss", None),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Stage 2 complete.")


if __name__ == "__main__":
    main()
