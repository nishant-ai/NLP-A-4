#!/bin/bash

################################################################################
# Final Push to >65% F1 Score
################################################################################

echo "=========================================="
echo "T5 Final Optimization (Target: >65% F1)"
echo "Start Time: $(date)"
echo "=========================================="

# Create directories
mkdir -p logs checkpoints results records

# Three strategies to push past 65%:

echo ""
echo "=========================================="
echo "Strategy 1: Continue training from best checkpoint"
echo "=========================================="

# Continue training from your best model with lower learning rate
python train_t5.py \
    --finetune \
    --learning_rate 1e-4 \
    --max_n_epochs 10 \
    --patience_epochs 3 \
    --batch_size 8 \
    --test_batch_size 16 \
    --scheduler_type linear \
    --num_warmup_epochs 1 \
    --weight_decay 0.01 \
    --experiment_name continue_best \
    2>&1 | tee logs/continue_training.log

echo ""
echo "=========================================="
echo "Strategy 2: Longer training with more patience"
echo "=========================================="

# Train for more epochs with higher patience
python train_t5.py \
    --finetune \
    --learning_rate 3e-4 \
    --max_n_epochs 30 \
    --patience_epochs 8 \
    --batch_size 8 \
    --test_batch_size 16 \
    --scheduler_type cosine \
    --num_warmup_epochs 3 \
    --weight_decay 0.01 \
    --experiment_name longer_training \
    2>&1 | tee logs/longer_training.log

echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "=========================================="