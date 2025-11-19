#!/bin/bash
#SBATCH --job-name=t5_scratch_ec
#SBATCH --account=ds_ga_1011_001-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --requeue

echo "=========================================="
echo "T5 Training from Scratch (Extra Credit)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Set environment variable to avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Change to working directory on scratch
cd /scratch/$USER/NLP-A4/hw4-code/part-2-code

# Create necessary directories
mkdir -p logs checkpoints results records

# Activate conda environment
source ~/.bashrc
conda activate hw4-part-2-nlp

# Run T5 training from scratch
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

# Copy results to home directory for easy retrieval
RESULTS_DIR=~/hw4_scr_results_$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULTS_DIR

echo "Copying results to: $RESULTS_DIR"

cp results/t5_scr_scr_experiment_*.sql $RESULTS_DIR/ 2>/dev/null
cp records/t5_scr_scr_experiment_*.pkl $RESULTS_DIR/ 2>/dev/null
cp checkpoints/scr_experiments/scr_experiment/best_model.pt $RESULTS_DIR/ 2>/dev/null

# Rename test files for submission format
if [ -f "$RESULTS_DIR/t5_scr_scr_experiment_test.sql" ]; then
    cp $RESULTS_DIR/t5_scr_scr_experiment_test.sql $RESULTS_DIR/t5_ft_experiment_ec_test.sql
    cp $RESULTS_DIR/t5_scr_scr_experiment_test.pkl $RESULTS_DIR/t5_ft_experiment_ec_test.pkl
    echo "Created submission files:"
    echo "  - t5_ft_experiment_ec_test.sql"
    echo "  - t5_ft_experiment_ec_test.pkl"
fi

echo ""
echo "Results copied to home directory: $RESULTS_DIR"
echo ""
echo "Files for Gradescope Extra Credit submission:"
ls -lh $RESULTS_DIR/*ec_test* 2>/dev/null || echo "Test files will be available after training completes"
echo ""
echo "=========================================="
echo "Job Complete"
echo "=========================================="
