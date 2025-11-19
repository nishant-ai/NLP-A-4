#!/bin/bash

# Generate submission files for Extra Credit question
# This creates the test set predictions required for hw4.pdf Extra Credit
# Usage: bash generate_ec_submission.sh [checkpoint_path]
#
# Examples:
#   bash generate_ec_submission.sh checkpoints/scr_experiments/scr_experiment/best_model.pt
#   bash generate_ec_submission.sh /scratch/$USER/NLP-A4/hw4-code/part-2-code/checkpoints/scr_experiments/scr_experiment/best_model.pt

echo "=========================================="
echo "Generating Extra Credit Submission Files"
echo "Start Time: $(date)"
echo "=========================================="

# Set environment variable to avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Path to your saved checkpoint from scratch training
# Can be provided as command line argument or use default
if [ -n "$1" ]; then
    CHECKPOINT_PATH="$1"
else
    # Default paths to try
    CHECKPOINT_PATH="checkpoints/scr_experiments/scr_experiment/best_model.pt"
fi

echo "Looking for checkpoint at: $CHECKPOINT_PATH"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    echo ""
    echo "Usage: bash generate_ec_submission.sh [checkpoint_path]"
    echo ""
    echo "Looking for checkpoints in current directory:"
    find checkpoints -name "*.pt" 2>/dev/null || echo "No checkpoints found in ./checkpoints"
    echo ""
    echo "Try running on HPC:"
    echo "  cd /scratch/\$USER/NLP-A4/hw4-code/part-2-code"
    echo "  bash generate_ec_submission.sh checkpoints/scr_experiments/scr_experiment/best_model.pt"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_PATH"
echo ""

# Create necessary directories
mkdir -p results records

# Output files as specified in hw4.pdf Extra Credit section
OUTPUT_SQL="results/t5_ft_experiment_ec_test.sql"
OUTPUT_RECORDS="records/t5_ft_experiment_ec_test.pkl"

echo "Generating test set predictions..."
echo "Output files:"
echo "  - $OUTPUT_SQL"
echo "  - $OUTPUT_RECORDS"
echo ""

# Run inference to generate test predictions
python generate_test_predictions.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --test_batch_size 16 \
    --output_sql "$OUTPUT_SQL" \
    --output_records "$OUTPUT_RECORDS"

echo ""
echo "=========================================="
echo "Submission Files Generated Successfully!"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "📦 Files ready for Gradescope submission:"
echo "  ✅ $OUTPUT_SQL"
echo "  ✅ $OUTPUT_RECORDS"
echo ""
echo "Upload these files to Gradescope for Extra Credit!"
