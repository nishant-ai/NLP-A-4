#!/bin/bash

################################################################################
# Optimized T5 Text-to-SQL Training Script
# Target: >65% F1 Score
################################################################################

echo "=========================================="
echo "T5 Optimized Training (>65% F1 Target)"
echo "Start Time: $(date)"
echo "=========================================="

# Create directories
mkdir -p logs checkpoints results records

# Print environment info
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Copy the optimized files to replace originals
echo "Using optimized implementations..."
cp /mnt/user-data/outputs/train_t5.py ./train_t5.py
cp /mnt/user-data/outputs/load_data.py ./load_data.py

################################################################################
# OPTIMAL CONFIGURATION FOR >65% F1
################################################################################

EXPERIMENT_NAME="optimal_t5"
OUTPUT_FILE="logs/optimal_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).txt"

echo ""
echo "=========================================="
echo "Configuration:"
echo "  - Model: T5-small (finetuned)"
echo "  - Learning Rate: 3e-4"
echo "  - Batch Size: 8"
echo "  - Max Epochs: 20"
echo "  - Patience: 5"
echo "  - Scheduler: Linear with warmup"
echo "  - Beam Search: 5 beams"
echo "=========================================="
echo ""

{
    echo "=========================================="
    echo "EXPERIMENT: ${EXPERIMENT_NAME}"
    echo "Start Time: $(date)"
    echo "=========================================="
    echo ""
    echo "KEY OPTIMIZATIONS:"
    echo "  1. Better prompt format (question: ... context: ...)"
    echo "  2. Proper label padding with -100"
    echo "  3. Beam search with 5 beams"
    echo "  4. Optimal learning rate (3e-4)"
    echo "  5. Gradient clipping"
    echo "  6. Simplified schema representation"
    echo ""
    echo "=========================================="
    echo "TRAINING OUTPUT:"
    echo "=========================================="
    echo ""
} > ${OUTPUT_FILE}

python train_t5.py \
    --finetune \
    --learning_rate 3e-4 \
    --max_n_epochs 20 \
    --patience_epochs 5 \
    --batch_size 8 \
    --test_batch_size 16 \
    --scheduler_type linear \
    --num_warmup_epochs 2 \
    --weight_decay 0.01 \
    --experiment_name ${EXPERIMENT_NAME} 2>&1 | tee -a ${OUTPUT_FILE}

{
    echo ""
    echo "=========================================="
    echo "TRAINING COMPLETED"
    echo "End Time: $(date)"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - Model: checkpoints/ft_experiments/${EXPERIMENT_NAME}/"
    echo "  - Results: results/t5_ft_ft_experiment_test.sql"
    echo "  - Records: records/t5_ft_ft_experiment_test.pkl"
    echo "  - Logs: ${OUTPUT_FILE}"
    echo "=========================================="
} >> ${OUTPUT_FILE}

echo "Training complete! Check ${OUTPUT_FILE} for details."