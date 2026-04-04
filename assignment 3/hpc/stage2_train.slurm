#!/bin/bash
#SBATCH --job-name=stage2_json
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@my.utsa.edu
#SBATCH --dependency=afterok:STAGE1_JOB_ID   # Replace STAGE1_JOB_ID with actual ID

# ============================================================
# Stage 2: JSON Instruct Fine-Tuning (UTSA HPC)
# Run AFTER stage1_train.slurm completes successfully.
# Submit with: sbatch --dependency=afterok:<stage1_jobid> hpc/stage2_train.slurm
# ============================================================

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Start time:   $(date)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# --- Environment setup ---
module load cuda/12.1
module load python/3.11

source ~/envs/llm_env/bin/activate

export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
export TOGETHER_API_KEY="your_key_here"

cd $SLURM_SUBMIT_DIR

echo ""
echo "============================================================"
echo "Stage 2: Constructing Teacher-Generated JSON Instruct Data"
echo "============================================================"
python data/construct_json_instruct.py --config config.yaml

echo ""
echo "============================================================"
echo "Stage 2: QLoRA Fine-Tuning on JSON Instruct Data"
echo "============================================================"
python training/stage2_json_instruct.py --config config.yaml

echo ""
echo "============================================================"
echo "Stage 2: Generating Inference Outputs (Checkpoint 2)"
echo "============================================================"
python inference/generate_outputs.py --config config.yaml --checkpoint 2

echo ""
echo "============================================================"
echo "Stage 2: Computing Automatic Metrics"
echo "============================================================"
python evaluation/json_metrics.py --config config.yaml --checkpoint 2

echo ""
echo "============================================================"
echo "Stage 2: Running Judge Evaluations"
echo "============================================================"
# Alpaca pairwise judge: 0 vs 1, 1 vs 2, 0 vs 2
python evaluation/judge_eval.py --config config.yaml --mode alpaca --ckpt-a 0 --ckpt-b 1
python evaluation/judge_eval.py --config config.yaml --mode alpaca --ckpt-a 1 --ckpt-b 2
python evaluation/judge_eval.py --config config.yaml --mode alpaca --ckpt-a 0 --ckpt-b 2

# JSON qualitative scoring at all checkpoints
python evaluation/judge_eval.py --config config.yaml --mode json --ckpt 0
python evaluation/judge_eval.py --config config.yaml --mode json --ckpt 1
python evaluation/judge_eval.py --config config.yaml --mode json --ckpt 2

echo ""
echo "============================================================"
echo "Aggregating All Results"
echo "============================================================"
python evaluation/aggregate_results.py --config config.yaml

echo ""
echo "Stage 2 pipeline complete."
echo "End time: $(date)"
