#!/bin/bash

################################################################################
# T5 Text-to-SQL Training Script (Fixed Version)
# Expected F1 score: 70-80% with proper training
################################################################################

# Print job information
echo "=========================================="
echo "Starting T5 Training - Fixed Version"
echo "Start Time: $(date)"
echo "=========================================="

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p results
mkdir -p records

# Print environment information
echo ""
echo "=========================================="
echo "Environment Information:"
echo "=========================================="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

################################################################################
# MAIN EXPERIMENT: Best Configuration for Text-to-SQL
################################################################################

EXPERIMENT_NAME="best_t5_sql"
OUTPUT_FILE="logs/results_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).txt"

echo "=========================================="
echo "Starting Experiment: ${EXPERIMENT_NAME}"
echo "=========================================="
echo ""

{
    echo "=========================================="
    echo "EXPERIMENT: ${EXPERIMENT_NAME}"
    echo "Start Time: $(date)"
    echo "=========================================="
    echo ""
    echo "HYPERPARAMETERS:"
    echo "  - Finetuning: Yes (CRITICAL)"
    echo "  - Learning Rate: 5e-4 (optimal for T5-small)"
    echo "  - Epochs: 15"
    echo "  - Patience: 5"
    echo "  - Batch Size: 16"
    echo "  - Test Batch Size: 32"
    echo "  - Scheduler: linear (works well with T5)"
    echo "  - Warmup Epochs: 2"
    echo "  - Weight Decay: 0.01"
    echo ""
    echo "=========================================="
    echo "TRAINING OUTPUT:"
    echo "=========================================="
    echo ""
} > ${OUTPUT_FILE}

# Run with the FIXED code
python train_t5_fixed.py \
    --finetune \
    --learning_rate 5e-4 \
    --max_n_epochs 15 \
    --patience_epochs 5 \
    --batch_size 16 \
    --test_batch_size 32 \
    --scheduler_type linear \
    --num_warmup_epochs 2 \
    --weight_decay 0.01 \
    --experiment_name ${EXPERIMENT_NAME} 2>&1 | tee -a ${OUTPUT_FILE}

{
    echo ""
    echo "=========================================="
    echo "EXPERIMENT COMPLETED"
    echo "End Time: $(date)"
    echo "=========================================="
} >> ${OUTPUT_FILE}

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Results saved in: logs/results_${EXPERIMENT_NAME}_*.txt"
echo "Checkpoint saved in: checkpoints/ft_experiments/${EXPERIMENT_NAME}/"
echo "Test predictions in: results/t5_ft_ft_experiment_test.sql"
echo "=========================================="

exit 0