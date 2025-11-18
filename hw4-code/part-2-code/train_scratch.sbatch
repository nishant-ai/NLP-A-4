#!/bin/bash

#SBATCH --job-name=t5_scratch
#SBATCH --account=ds_ga_1011_001-2025fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

################################################################################
# T5 Training from Scratch - SLURM Batch Job
# Target: >=50% F1 Score (Extra Credit)
################################################################################

echo "=========================================="
echo "T5 Training from Scratch (SLURM)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Create directories
mkdir -p logs checkpoints results records

# Set environment variable to avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Activate your conda environment
# Modify this path based on your setup
source ~/.bashrc
conda activate hw4-part-2-nlp

# Navigate to project directory (modify as needed)
cd $HOME/hw4-code/part-2-code

################################################################################
# Step 1: Data Augmentation
################################################################################

echo ""
echo "=========================================="
echo "Step 1: Running Data Augmentation"
echo "=========================================="

python augment_data.py

# Check if augmentation was successful
if [ ! -f "data/train_augmented.nl" ] || [ ! -f "data/train_augmented.sql" ]; then
    echo "ERROR: Data augmentation failed! Files not found."
    exit 1
fi

echo "Data augmentation complete!"

################################################################################
# Step 2: Training from Scratch
################################################################################

echo ""
echo "=========================================="
echo "Step 2: Training T5 from Scratch"
echo "=========================================="

# Strategy 1: Higher LR with Linear scheduler
echo "Running Strategy 1: Linear scheduler"
python train_t5.py \
    --learning_rate 5e-4 \
    --max_n_epochs 50 \
    --patience_epochs 10 \
    --batch_size 8 \
    --test_batch_size 16 \
    --scheduler_type linear \
    --num_warmup_epochs 5 \
    --weight_decay 0.01 \
    --experiment_name scratch_ec

################################################################################
# Step 3: Rename output files for submission
################################################################################

echo ""
echo "=========================================="
echo "Step 3: Preparing submission files"
echo "=========================================="

# Rename files to match required submission format
if [ -f "results/t5_scr_ft_experiment_test.sql" ]; then
    cp results/t5_scr_ft_experiment_test.sql results/t5_ft_experiment_ec_test.sql
    echo "Created: results/t5_ft_experiment_ec_test.sql"
fi

if [ -f "records/t5_scr_ft_experiment_test.pkl" ]; then
    cp records/t5_scr_ft_experiment_test.pkl records/t5_ft_experiment_ec_test.pkl
    echo "Created: records/t5_ft_experiment_ec_test.pkl"
fi

echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "=========================================="

echo ""
echo "Submission files for Extra Credit:"
echo "  - results/t5_ft_experiment_ec_test.sql"
echo "  - records/t5_ft_experiment_ec_test.pkl"
