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
    --max_n_epochs 25 \
    --patience_epochs 6 \
    --batch_size 8 \
    --test_batch_size 16 \
    --scheduler_type cosine \
    --num_warmup_epochs 3 \
    --weight_decay 0.01 \
    --experiment_name run_t5_nov18_1 \
    2>&1 | tee logs/continue_training.log

echo ""
echo "=========================================="
echo "Strategy 2: Higher LR with Linear scheduler (like nov17_2)"
echo "=========================================="

# Use the successful nov17_2 configuration but with longer training
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
    --experiment_name strategy2_linear \
    2>&1 | tee logs/strategy2_linear.log

echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "=========================================="