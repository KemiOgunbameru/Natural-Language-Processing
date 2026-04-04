#!/bin/bash
#SBATCH --job-name=stage1_alpaca
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@my.utsa.edu

# ============================================================
# Stage 1: Alpaca Fine-Tuning (UTSA HPC)
# ============================================================

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Start time:   $(date)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# --- Environment setup ---
module load cuda/12.1
module load python/3.11

# Activate your virtual environment (update path as needed)
source ~/envs/llm_env/bin/activate

# Hugging Face cache (use scratch or project storage on HPC)
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false

# Set your Together AI key if using API-based teacher/judge
# export TOGETHER_API_KEY="your_key_here"

# Navigate to project root
cd $SLURM_SUBMIT_DIR

echo ""
echo "============================================================"
echo "Stage 1: Alpaca Data Preparation"
echo "============================================================"
python data/prepare_alpaca.py --config config.yaml

echo ""
echo "============================================================"
echo "Stage 1: QLoRA Fine-Tuning on Alpaca Data"
echo "============================================================"
python training/stage1_alpaca.py --config config.yaml

echo ""
echo "============================================================"
echo "Stage 1: Generating Inference Outputs (Checkpoint 0 & 1)"
echo "============================================================"
# Generate for both untuned (0) and stage1 (1) so eval can run
python inference/generate_outputs.py --config config.yaml --checkpoint 0
python inference/generate_outputs.py --config config.yaml --checkpoint 1

echo ""
echo "============================================================"
echo "Stage 1: Computing Automatic Metrics"
echo "============================================================"
python evaluation/json_metrics.py --config config.yaml --checkpoint 0
python evaluation/json_metrics.py --config config.yaml --checkpoint 1

echo ""
echo "Stage 1 pipeline complete."
echo "End time: $(date)"
