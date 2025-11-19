#!/bin/bash

# Run training from scratch experiment on compute node (Extra Credit)
# Usage: bash run_scratch.sh

echo "=========================================="
echo "T5 Training from Scratch (Extra Credit)"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variable to avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Change to working directory
cd /scratch/$USER/NLP-A4/hw4-code/part-2-code

# Create necessary directories
mkdir -p logs checkpoints results records

# Run T5 training from scratch
python train_t5.py \
    --learning_rate 5e-4 \
    --batch_size 8 \
    --test_batch_size 16 \
    --max_n_epochs 30 \
    --patience_epochs 7 \
    --scheduler_type cosine \
    --num_warmup_epochs 3 \
    --weight_decay 0.01 \
    --experiment_name scr_experiment

echo "=========================================="
echo "Training Complete"
echo "End Time: $(date)"
echo "=========================================="

# Copy results to home directory
RESULTS_DIR=~/hw4_scr_results_$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULTS_DIR

cp results/t5_scr_scr_experiment_*.sql $RESULTS_DIR/ 2>/dev/null
cp records/t5_scr_scr_experiment_*.pkl $RESULTS_DIR/ 2>/dev/null
cp checkpoints/scr_experiments/scr_experiment/best_model.pt $RESULTS_DIR/ 2>/dev/null

# Rename for submission format
cp $RESULTS_DIR/t5_scr_scr_experiment_test.sql $RESULTS_DIR/t5_ft_experiment_ec_test.sql 2>/dev/null
cp $RESULTS_DIR/t5_scr_scr_experiment_test.pkl $RESULTS_DIR/t5_ft_experiment_ec_test.pkl 2>/dev/null

echo "Results copied to: $RESULTS_DIR"
echo ""
echo "Files for Gradescope Extra Credit submission:"
ls -la $RESULTS_DIR/*ec_test* 2>/dev/null || echo "No test files found"
