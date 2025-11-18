#!/bin/bash

################################################################################
# T5 Training from Scratch (Extra Credit)
# Target: >=50% F1 Score
################################################################################

echo "=========================================="
echo "T5 Training from Scratch (Extra Credit)"
echo "Start Time: $(date)"
echo "=========================================="

# Create directories
mkdir -p logs checkpoints results records

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
echo ""

################################################################################
# Step 2: Training from Scratch with Augmented Data
################################################################################

echo "=========================================="
echo "Step 2: Training T5 from Scratch"
echo "=========================================="

echo ""
echo "Strategy 1: Higher LR with Linear scheduler"
echo "=========================================="

# Note: NO --finetune flag = training from scratch
python train_t5.py \
    --learning_rate 5e-4 \
    --max_n_epochs 50 \
    --patience_epochs 10 \
    --batch_size 8 \
    --test_batch_size 16 \
    --scheduler_type linear \
    --num_warmup_epochs 5 \
    --weight_decay 0.01 \
    --experiment_name scratch_ec \
    2>&1 | tee logs/scratch_linear.log

echo ""
echo "=========================================="
echo "Strategy 2: Cosine scheduler with warmup"
echo "=========================================="

python train_t5.py \
    --learning_rate 3e-4 \
    --max_n_epochs 50 \
    --patience_epochs 10 \
    --batch_size 8 \
    --test_batch_size 16 \
    --scheduler_type cosine \
    --num_warmup_epochs 7 \
    --weight_decay 0.01 \
    --experiment_name scratch_cosine \
    2>&1 | tee logs/scratch_cosine.log

################################################################################
# Step 3: Rename output files for submission
################################################################################

echo ""
echo "=========================================="
echo "Step 3: Preparing submission files"
echo "=========================================="

# Rename files to match required submission format
# From: t5_scr_ft_experiment_test.* -> To: t5_ft_experiment_ec_test.*
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
echo "=========================================="
