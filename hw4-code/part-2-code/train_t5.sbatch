#!/bin/bash
#SBATCH --job-name=t5_ft_train
#SBATCH --account=ds_ga_1011_001-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/t5_ft_%j.out
#SBATCH --error=logs/t5_ft_%j.err

# Print job info
echo "=========================================="
echo "T5 Fine-tuning for Text-to-SQL"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p checkpoints
mkdir -p results
mkdir -p records

# Load conda environment
module purge
module load anaconda3/2020.07

# Activate your conda environment
source ~/.bashrc
conda activate hw4-part-2-nlp

# Set environment variable to avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Change to working directory
cd /scratch/$USER/NLP-A4/hw4-code/part-2-code

# Run T5 fine-tuning
echo "=========================================="
echo "Starting T5 Fine-tuning"
echo "=========================================="

python train_t5.py \
    --finetune \
    --learning_rate 1e-4 \
    --batch_size 8 \
    --test_batch_size 16 \
    --max_n_epochs 20 \
    --patience_epochs 5 \
    --scheduler_type cosine \
    --num_warmup_epochs 2 \
    --weight_decay 0.01 \
    --experiment_name t5_ft_experiment

echo "=========================================="
echo "Training Complete"
echo "End Time: $(date)"
echo "=========================================="

# Copy results to home directory for easy access
echo "=========================================="
echo "Copying results to home directory..."
echo "=========================================="

RESULTS_DIR=~/hw4_results_$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULTS_DIR

# Copy output files
cp -r results/*.sql $RESULTS_DIR/ 2>/dev/null || echo "No SQL files found"
cp -r records/*.pkl $RESULTS_DIR/ 2>/dev/null || echo "No PKL files found"
cp -r checkpoints/ft_experiments/t5_ft_experiment/*.pt $RESULTS_DIR/ 2>/dev/null || echo "No checkpoint files found"

echo "Results copied to: $RESULTS_DIR"
echo ""
echo "Files for Gradescope submission:"
ls -la $RESULTS_DIR/*test* 2>/dev/null || echo "No test files found"
echo ""
echo "=========================================="
echo "Job Complete!"
echo "=========================================="
