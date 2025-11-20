#!/bin/bash
##########################################
# T5 Training from Scratch - Simple Version
# Assumes:
#   - You're already in part-2-code directory
#   - Conda environment is activated
#   - On a compute node with GPU
##########################################

echo "=========================================="
echo "T5 Training from Scratch (Extra Credit)"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variable to avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Verify we're in the right place
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Create necessary directories
mkdir -p logs checkpoints results records

# Run T5 training from scratch
echo "Starting training with following parameters:"
echo "  - Learning rate: 1e-4"
echo "  - Batch size: 8 (train), 16 (test)"
echo "  - Max epochs: 30"
echo "  - Patience: 10 epochs"
echo "  - Warmup: 5 epochs"
echo ""

python train_t5.py \
    --learning_rate 1e-4 \
    --batch_size 8 \
    --test_batch_size 16 \
    --max_n_epochs 30 \
    --patience_epochs 10 \
    --scheduler_type cosine \
    --num_warmup_epochs 5 \
    --weight_decay 0.01 \
    --experiment_name scr_experiment

echo ""
echo "=========================================="
echo "Training Complete"
echo "End Time: $(date)"
echo "=========================================="

# Show generated files
echo ""
echo "Generated files:"
echo "Dev set:"
ls -lh results/t5_scr_scr_experiment_dev.sql 2>/dev/null
ls -lh records/t5_scr_scr_experiment_dev.pkl 2>/dev/null
echo ""
echo "Test set:"
ls -lh results/t5_scr_scr_experiment_test.sql 2>/dev/null
ls -lh records/t5_scr_scr_experiment_test.pkl 2>/dev/null
echo ""
echo "Checkpoints:"
ls -lh checkpoints/scr_experiments/scr_experiment/*.pt 2>/dev/null

echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Copy results to home directory:"
echo "   mkdir -p ~/hw4_scr_results"
echo "   cp results/t5_scr_scr_experiment_test.* ~/hw4_scr_results/"
echo "   cp records/t5_scr_scr_experiment_test.* ~/hw4_scr_results/"
echo "   cp checkpoints/scr_experiments/scr_experiment/best_model.pt ~/hw4_scr_results/"
echo ""
echo "2. Download to local machine:"
echo "   scp -r \$USER@greene.hpc.nyu.edu:~/hw4_scr_results /path/to/local"
echo "=========================================="
