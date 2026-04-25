# Assignment 3 — Sequential Instruction Tuning of a Small LLM

**LLM & Agentic Systems | Graduate Course | UTSA**
**Dr. Peyman Najafirad (Paul Rad) | TA: Mohammad Bahrami**

---

## Overview

This repository implements a complete two-stage instruction-tuning pipeline for
**Phi-3.5 Mini Instruct** using QLoRA on UTSA HPC. The pipeline investigates
catastrophic forgetting: does a second fine-tuning stage on structured JSON data
degrade the general instruction-following ability gained from Stage 1 Alpaca training?

### Pipeline

```
Base Model (Phi-3.5 Mini)
        │
        ▼  Stage 1: QLoRA on Alpaca data
Checkpoint 1 (Alpaca-tuned)
        │
        ▼  Stage 2: QLoRA on teacher-generated JSON Instruct data
Checkpoint 2 (JSON-tuned)
        │
        ▼  Judge + automatic metrics at all 3 checkpoints
Forgetting Analysis Report
```

---

## Repository Structure

```
assignment3/
├── config.yaml                          # All hyperparameters and paths
├── requirements.txt
├── data/
│   ├── prepare_alpaca.py                # Download and clean Alpaca dataset
│   └── construct_json_instruct.py       # Imitation learning pipeline
├── training/
│   ├── stage1_alpaca.py                 # Stage 1 QLoRA fine-tuning
│   └── stage2_json_instruct.py          # Stage 2 QLoRA fine-tuning
├── inference/
│   └── generate_outputs.py             # Generate responses at all 3 checkpoints
├── evaluation/
│   ├── judge_eval.py                   # LLM-as-a-Judge (pairwise + qualitative)
│   ├── json_metrics.py                 # JSON validity, F1, ROUGE, BERTScore
│   └── aggregate_results.py            # Build final comparison table
├── prompts/
│   ├── teacher_generation/             # Templates for teacher data generation
│   └── judge/                          # Templates for judge evaluation
├── hpc/
│   ├── stage1_train.slurm              # SLURM job: Stage 1
│   └── stage2_train.slurm              # SLURM job: Stage 2
├── outputs/
│   ├── checkpoint0/                    # Untuned base model responses
│   ├── checkpoint1/                    # Stage 1 adapter + responses
│   └── checkpoint2/                    # Stage 2 adapter + responses
├── logs/                               # Training logs, judge scores, summaries
└── REPORT.md                           # GitHub blog post (full 5-page report)
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/assignment3-llm-finetuning.git
cd assignment3-llm-finetuning
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
export TOGETHER_API_KEY="your_together_ai_key_here"
export HUGGING_FACE_HUB_TOKEN="your_hf_token_here"
```

> **Phi-3.5 Mini access:** Accept the model license at
> https://huggingface.co/microsoft/Phi-3.5-mini-instruct then run
> `huggingface-cli login`.

---

## Reproducing the Full Pipeline

### Local (step by step)

```bash
python data/prepare_alpaca.py --config config.yaml
python data/construct_json_instruct.py --config config.yaml
python training/stage1_alpaca.py --config config.yaml
python training/stage2_json_instruct.py --config config.yaml
python inference/generate_outputs.py --config config.yaml --checkpoint all
python evaluation/json_metrics.py --config config.yaml --checkpoint all
python evaluation/judge_eval.py --config config.yaml --mode alpaca --ckpt-a 0 --ckpt-b 1
python evaluation/judge_eval.py --config config.yaml --mode alpaca --ckpt-a 1 --ckpt-b 2
python evaluation/judge_eval.py --config config.yaml --mode alpaca --ckpt-a 0 --ckpt-b 2
python evaluation/judge_eval.py --config config.yaml --mode json --ckpt 0
python evaluation/judge_eval.py --config config.yaml --mode json --ckpt 1
python evaluation/judge_eval.py --config config.yaml --mode json --ckpt 2
python evaluation/aggregate_results.py --config config.yaml
```

### UTSA HPC (recommended)

```bash
# Submit Stage 1
sbatch hpc/stage1_train.slurm

# Submit Stage 2 with dependency on Stage 1 completing
sbatch --dependency=afterok:<STAGE1_JOB_ID> hpc/stage2_train.slurm

# Monitor
squeue -u $USER
tail -f logs/stage1_<jobid>.out
```

---

## Key Design Decisions

**Why Phi-3.5 Mini?** Best instruction-following performance in the sub-4B
parameter class; fits cleanly in a single A100 under 4-bit QLoRA; uses a
well-defined chat template compatible with this pipeline.

**Why QLoRA?** Enables single-GPU fine-tuning via 4-bit NF4 quantisation + LoRA
adapter training (~0.5% of total parameters). Both stages run within UTSA HPC
time limits on one A100.

**Stage 2 adapter strategy:** Stage 1 LoRA is merged into base weights via
`merge_and_unload()` before attaching a fresh Stage 2 LoRA. This gives Stage 2
a clean post-Alpaca starting point without adapter-stacking complexity.

**Imitation learning vs distillation:** Teacher outputs are used as text-level
supervision (cross-entropy loss), not logit-level KL divergence. This is
black-box distillation / synthetic data generation — the student only ever sees
the teacher's final text responses.

---

## References

1. Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
2. Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
3. Taori et al. (2023). Alpaca: A Strong, Replicable Instruction-Following Model.
4. Wang et al. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions.
5. Gu et al. (2024). A Survey on LLM-as-a-Judge.
