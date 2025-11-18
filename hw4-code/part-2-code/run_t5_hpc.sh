#!/bin/bash
#SBATCH --job-name=t5_text2sql
#SBATCH --output=logs/t5_%j.out
#SBATCH --error=logs/t5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_netid@nyu.edu

################################################################################
# T5 Text-to-SQL Training on NYU HPC
# Expected runtime: 1-2 hours with A100
# Expected F1 score: 70-80%
################################################################################

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p results
mkdir -p records

# Load required modules (adjust based on your HPC environment)
module purge
module load python/3.9
module load cuda/11.8
module load cudnn/8.6.0

# Activate virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Or use conda
# module load anaconda3
# conda activate your_env

# Print Python and CUDA info
echo ""
echo "=========================================="
echo "Environment Information:"
echo "=========================================="
python --version
which python
nvcc --version
nvidia-smi
echo ""

# Print PyTorch and GPU info
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

################################################################################
# EXPERIMENT 1: Recommended Configuration (Most Likely to Succeed)
################################################################################

EXPERIMENT_NAME="recommended_ft"
OUTPUT_FILE="logs/results_${EXPERIMENT_NAME}_${SLURM_JOB_ID}.txt"

echo "=========================================="
echo "Starting Experiment: ${EXPERIMENT_NAME}"
echo "=========================================="
echo ""

{
    echo "=========================================="
    echo "EXPERIMENT: ${EXPERIMENT_NAME}"
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Start Time: $(date)"
    echo "=========================================="
    echo ""
    echo "HYPERPARAMETERS:"
    echo "  - Finetuning: Yes"
    echo "  - Learning Rate: 3e-4"
    echo "  - Epochs: 8"
    echo "  - Patience: 4"
    echo "  - Batch Size: 8"
    echo "  - Test Batch Size: 16"
    echo "  - Scheduler: cosine"
    echo "  - Warmup Epochs: 1"
    echo "  - Weight Decay: 0.01"
    echo ""
    echo "=========================================="
    echo "TRAINING OUTPUT:"
    echo "=========================================="
    echo ""
} > ${OUTPUT_FILE}

python train_t5.py \
    --finetune \
    --learning_rate 3e-4 \
    --max_n_epochs 8 \
    --patience_epochs 4 \
    --batch_size 8 \
    --test_batch_size 16 \
    --scheduler_type cosine \
    --num_warmup_epochs 1 \
    --weight_decay 0.01 \
    --experiment_name ${EXPERIMENT_NAME} 2>&1 | tee -a ${OUTPUT_FILE}

{
    echo ""
    echo "=========================================="
    echo "EXPERIMENT COMPLETED"
    echo "End Time: $(date)"
    echo "=========================================="
    echo ""
} >> ${OUTPUT_FILE}

################################################################################
# EXPERIMENT 2: Optimized Configuration (Higher LR, more aggressive)
################################################################################

EXPERIMENT_NAME="optimized_ft"
OUTPUT_FILE="logs/results_${EXPERIMENT_NAME}_${SLURM_JOB_ID}.txt"

echo ""
echo "=========================================="
echo "Starting Experiment: ${EXPERIMENT_NAME}"
echo "=========================================="
echo ""

{
    echo "=========================================="
    echo "EXPERIMENT: ${EXPERIMENT_NAME}"
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Start Time: $(date)"
    echo "=========================================="
    echo ""
    echo "HYPERPARAMETERS:"
    echo "  - Finetuning: Yes"
    echo "  - Learning Rate: 5e-4"
    echo "  - Epochs: 10"
    echo "  - Patience: 5"
    echo "  - Batch Size: 8"
    echo "  - Test Batch Size: 16"
    echo "  - Scheduler: cosine"
    echo "  - Warmup Epochs: 2"
    echo "  - Weight Decay: 0.01"
    echo ""
    echo "=========================================="
    echo "TRAINING OUTPUT:"
    echo "=========================================="
    echo ""
} > ${OUTPUT_FILE}

python train_t5.py \
    --finetune \
    --learning_rate 5e-4 \
    --max_n_epochs 10 \
    --patience_epochs 5 \
    --batch_size 8 \
    --test_batch_size 16 \
    --scheduler_type cosine \
    --num_warmup_epochs 2 \
    --weight_decay 0.01 \
    --experiment_name ${EXPERIMENT_NAME} 2>&1 | tee -a ${OUTPUT_FILE}

{
    echo ""
    echo "=========================================="
    echo "EXPERIMENT COMPLETED"
    echo "End Time: $(date)"
    echo "=========================================="
    echo ""
} >> ${OUTPUT_FILE}

################################################################################
# EXPERIMENT 3: Conservative Configuration (Safer, more stable)
################################################################################

EXPERIMENT_NAME="conservative_ft"
OUTPUT_FILE="logs/results_${EXPERIMENT_NAME}_${SLURM_JOB_ID}.txt"

echo ""
echo "=========================================="
echo "Starting Experiment: ${EXPERIMENT_NAME}"
echo "=========================================="
echo ""

{
    echo "=========================================="
    echo "EXPERIMENT: ${EXPERIMENT_NAME}"
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Start Time: $(date)"
    echo "=========================================="
    echo ""
    echo "HYPERPARAMETERS:"
    echo "  - Finetuning: Yes"
    echo "  - Learning Rate: 1e-4"
    echo "  - Epochs: 12"
    echo "  - Patience: 5"
    echo "  - Batch Size: 16"
    echo "  - Test Batch Size: 32"
    echo "  - Scheduler: linear"
    echo "  - Warmup Epochs: 1"
    echo "  - Weight Decay: 0.01"
    echo ""
    echo "=========================================="
    echo "TRAINING OUTPUT:"
    echo "=========================================="
    echo ""
} > ${OUTPUT_FILE}

python train_t5.py \
    --finetune \
    --learning_rate 1e-4 \
    --max_n_epochs 12 \
    --patience_epochs 5 \
    --batch_size 16 \
    --test_batch_size 32 \
    --scheduler_type linear \
    --num_warmup_epochs 1 \
    --weight_decay 0.01 \
    --experiment_name ${EXPERIMENT_NAME} 2>&1 | tee -a ${OUTPUT_FILE}

{
    echo ""
    echo "=========================================="
    echo "EXPERIMENT COMPLETED"
    echo "End Time: $(date)"
    echo "=========================================="
    echo ""
} >> ${OUTPUT_FILE}

################################################################################
# Summary and Cleanup
################################################################################

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - logs/results_recommended_ft_${SLURM_JOB_ID}.txt"
echo "  - logs/results_optimized_ft_${SLURM_JOB_ID}.txt"
echo "  - logs/results_conservative_ft_${SLURM_JOB_ID}.txt"
echo ""
echo "Checkpoints saved in:"
echo "  - checkpoints/ft_experiments/recommended_ft/"
echo "  - checkpoints/ft_experiments/optimized_ft/"
echo "  - checkpoints/ft_experiments/conservative_ft/"
echo ""
echo "Test predictions saved in:"
echo "  - results/t5_ft_ft_experiment_test.sql"
echo "  - records/t5_ft_ft_experiment_test.pkl"
echo ""
echo "Job End Time: $(date)"
echo "=========================================="

# Print GPU utilization summary
nvidia-smi

exit 0
