#!/bin/bash

# Run both finetuning and scratch experiments
# Execute from compute node: bash run_experiments.sh

echo "=========================================="
echo "Running T5 Experiments"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variable to avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p results
mkdir -p records

# ==========================================
# Experiment 1: Fine-tuning (Q7)
# ==========================================
echo ""
echo "=========================================="
echo "Experiment 1: Fine-tuning T5"
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

echo ""
echo "Fine-tuning complete!"
echo "=========================================="

# ==========================================
# Experiment 2: Training from Scratch (Extra Credit)
# ==========================================
echo ""
echo "=========================================="
echo "Experiment 2: Training T5 from Scratch (Extra Credit)"
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

echo ""
echo "Training from scratch complete!"
echo "=========================================="

# ==========================================
# Copy results
# ==========================================
echo ""
echo "=========================================="
echo "Copying results to home directory..."
echo "=========================================="

RESULTS_DIR=~/hw4_results_$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULTS_DIR

# Copy finetuning results
cp results/t5_ft_ft_experiment_*.sql $RESULTS_DIR/ 2>/dev/null
cp records/t5_ft_ft_experiment_*.pkl $RESULTS_DIR/ 2>/dev/null
cp checkpoints/ft_experiments/t5_ft_experiment/*.pt $RESULTS_DIR/ 2>/dev/null

# Copy scratch results (extra credit)
cp results/t5_scr_scr_experiment_*.sql $RESULTS_DIR/ 2>/dev/null
cp records/t5_scr_scr_experiment_*.pkl $RESULTS_DIR/ 2>/dev/null
cp checkpoints/scr_experiments/t5_scr_experiment/*.pt $RESULTS_DIR/ 2>/dev/null

echo "Results copied to: $RESULTS_DIR"
echo ""
echo "=========================================="
echo "Files for Gradescope submission:"
echo "=========================================="
echo ""
echo "Q7 (Fine-tuning):"
ls -la $RESULTS_DIR/t5_ft_*test* 2>/dev/null || echo "No finetuning test files found"
echo ""
echo "Extra Credit (From Scratch):"
ls -la $RESULTS_DIR/t5_scr_*test* 2>/dev/null || echo "No scratch test files found"
echo ""
echo "=========================================="
echo "All Experiments Complete!"
echo "End Time: $(date)"
echo "=========================================="
