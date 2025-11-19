#!/bin/bash
#SBATCH --job-name=t5_scr_train
#SBATCH --account=ds_ga_1011_001-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/t5_scr_%j.out
#SBATCH --error=logs/t5_scr_%j.err

# Print job info
echo "=========================================="
echo "T5 Training from Scratch (Extra Credit)"
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

# Clone or update code from GitHub
echo "=========================================="
echo "Pulling latest code from GitHub..."
echo "=========================================="

REPO_DIR=/scratch/$USER/NLP-A4
REPO_URL="https://github.com/nishant-ai/NLP-A-4"

if [ -d "$REPO_DIR" ]; then
    # Repository exists, pull latest changes
    cd $REPO_DIR
    git fetch --all
    git checkout inspired
    git reset --hard origin/inspired
    git pull origin inspired
else
    # Clone repository
    mkdir -p /scratch/$USER
    cd /scratch/$USER
    git clone -b inspired $REPO_URL NLP-A4
fi

# Change to working directory
cd /scratch/$USER/NLP-A4/hw4-code/part-2-code

# Show current git commit for verification
echo "Current Git Commit: $(git rev-parse --short HEAD)"
echo "Branch: $(git branch --show-current)"

# Run T5 training from scratch
echo "=========================================="
echo "Starting T5 Training from Scratch"
echo "=========================================="

python train_t5.py \
    --learning_rate 5e-4 \
    --batch_size 8 \
    --test_batch_size 16 \
    --max_n_epochs 30 \
    --patience_epochs 7 \
    --scheduler_type cosine \
    --num_warmup_epochs 3 \
    --weight_decay 0.01 \
    --experiment_name t5_scr_experiment

echo "=========================================="
echo "Training Complete"
echo "End Time: $(date)"
echo "=========================================="

# Copy results to home directory for easy access
echo "=========================================="
echo "Copying results to home directory..."
echo "=========================================="

RESULTS_DIR=~/hw4_scratch_results_$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULTS_DIR

# Copy output files
cp -r results/t5_scr_*.sql $RESULTS_DIR/ 2>/dev/null || echo "No SQL files found"
cp -r records/t5_scr_*.pkl $RESULTS_DIR/ 2>/dev/null || echo "No PKL files found"
cp -r checkpoints/scr_experiments/t5_scr_experiment/*.pt $RESULTS_DIR/ 2>/dev/null || echo "No checkpoint files found"

# Also create the required submission filenames
cp results/t5_scr_scr_experiment_test.sql $RESULTS_DIR/t5_ft_experiment_ec_test.sql 2>/dev/null
cp records/t5_scr_scr_experiment_test.pkl $RESULTS_DIR/t5_ft_experiment_ec_test.pkl 2>/dev/null

echo "Results copied to: $RESULTS_DIR"
echo ""
echo "Files for Gradescope Extra Credit submission:"
ls -la $RESULTS_DIR/*test* 2>/dev/null || echo "No test files found"
echo ""
echo "=========================================="
echo "Job Complete!"
echo "=========================================="
