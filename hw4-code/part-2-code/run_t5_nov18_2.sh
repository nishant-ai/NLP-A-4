#!/bin/bash

################################################################################
# T5 Training with Data Augmentation
# Target: >65% F1 Score
################################################################################

echo "=========================================="
echo "T5 Training with Data Augmentation"
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
# Step 2: Training with Augmented Data
################################################################################

echo "=========================================="
echo "Step 2: Training with Augmented Data"
echo "=========================================="

echo ""
echo "Strategy 1: Lower LR with Cosine scheduler"
echo "=========================================="

python train_t5.py \
    --finetune \
    --learning_rate 1e-4 \
    --max_n_epochs 25 \
    --patience_epochs 6 \
    --batch_size 8 \
    --test_batch_size 16 \
    --scheduler_type cosine \
    --num_warmup_epochs 3 \
    --weight_decay 0.01 \
    --experiment_name augmented_cosine \
    2>&1 | tee logs/augmented_cosine.log

echo ""
echo "=========================================="
echo "Strategy 2: Higher LR with Linear scheduler"
echo "=========================================="

python train_t5.py \
    --finetune \
    --learning_rate 3e-4 \
    --max_n_epochs 25 \
    --patience_epochs 7 \
    --batch_size 8 \
    --test_batch_size 16 \
    --scheduler_type linear \
    --num_warmup_epochs 2 \
    --weight_decay 0.01 \
    --experiment_name augmented_linear \
    2>&1 | tee logs/augmented_linear.log

echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "=========================================="
