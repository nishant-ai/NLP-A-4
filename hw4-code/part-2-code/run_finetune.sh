#!/bin/bash

# Run finetuning experiment on compute node
# Usage: bash run_finetune.sh

echo "=========================================="
echo "T5 Fine-tuning (Q7)"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variable to avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Change to working directory
cd /scratch/$USER/NLP-A4/hw4-code/part-2-code

# Create necessary directories
mkdir -p logs checkpoints results records

# Run T5 fine-tuning
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
    --experiment_name ft_experiment

echo "=========================================="
echo "Training Complete"
echo "End Time: $(date)"
echo "=========================================="

# Copy results to home directory
RESULTS_DIR=~/hw4_ft_results_$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULTS_DIR

cp results/t5_ft_ft_experiment_*.sql $RESULTS_DIR/ 2>/dev/null
cp records/t5_ft_ft_experiment_*.pkl $RESULTS_DIR/ 2>/dev/null
cp checkpoints/ft_experiments/ft_experiment/best_model.pt $RESULTS_DIR/ 2>/dev/null

echo "Results copied to: $RESULTS_DIR"
echo ""
echo "Files for Gradescope Q7 submission:"
ls -la $RESULTS_DIR/*test* 2>/dev/null || echo "No test files found"
