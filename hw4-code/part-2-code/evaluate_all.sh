#!/bin/bash

# Script to evaluate all model predictions using evaluate.py
# Usage: bash evaluate_all.sh

echo "=========================================="
echo "Evaluating All Models"
echo "=========================================="

# Ground truth paths
GT_DEV_SQL="data/dev.sql"
GT_DEV_RECORDS="records/ground_truth_dev.pkl"

# Check if ground truth files exist
if [ ! -f "$GT_DEV_SQL" ] || [ ! -f "$GT_DEV_RECORDS" ]; then
    echo "ERROR: Ground truth files not found!"
    echo "  Expected: $GT_DEV_SQL"
    echo "  Expected: $GT_DEV_RECORDS"
    exit 1
fi

echo ""

# Evaluate T5 Finetuned model (ft_experiment)
echo "=========================================="
echo "1. T5 Finetuned (ft_experiment) - DEV SET"
echo "=========================================="
if [ -f "results/t5_ft_ft_experiment_dev.sql" ] && [ -f "records/t5_ft_ft_experiment_dev.pkl" ]; then
    python evaluate.py \
        -ps results/t5_ft_ft_experiment_dev.sql \
        -pr records/t5_ft_ft_experiment_dev.pkl \
        -ds "$GT_DEV_SQL" \
        -dr "$GT_DEV_RECORDS"
else
    echo "⚠️  Files not found, skipping..."
fi

echo ""

# Evaluate T5 Scratch model (scr_experiment) if exists
echo "=========================================="
echo "2. T5 Scratch (scr_ft_experiment) - DEV SET"
echo "=========================================="
if [ -f "results/t5_scr_ft_experiment_dev.sql" ] && [ -f "records/t5_scr_ft_experiment_dev.pkl" ]; then
    python evaluate.py \
        -ps results/t5_scr_ft_experiment_dev.sql \
        -pr records/t5_scr_ft_experiment_dev.pkl \
        -ds "$GT_DEV_SQL" \
        -dr "$GT_DEV_RECORDS"
else
    echo "⚠️  Files not found, skipping..."
fi

echo ""

# Evaluate T5 ft_test
echo "=========================================="
echo "3. T5 Finetuned (t5_ft) - DEV SET"
echo "=========================================="
if [ -f "results/t5_ft_test.sql" ] && [ -f "records/t5_ft_test.pkl" ]; then
    # Note: These seem to be test files but we can only evaluate on dev
    echo "⚠️  These are test predictions - cannot evaluate without ground truth"
    echo "    Files: results/t5_ft_test.sql, records/t5_ft_test.pkl"
else
    echo "⚠️  Files not found, skipping..."
fi

echo ""

# Evaluate T5 scr_test
echo "=========================================="
echo "4. T5 Scratch (t5_scr) - TEST SET"
echo "=========================================="
if [ -f "results/t5_scr_test.sql" ] && [ -f "records/t5_scr_test.pkl" ]; then
    echo "⚠️  These are test predictions - cannot evaluate without ground truth"
    echo "    Files: results/t5_scr_test.sql, records/t5_scr_test.pkl"
else
    echo "⚠️  Files not found, skipping..."
fi

echo ""

# Evaluate LLM model if exists
echo "=========================================="
echo "5. LLM Model - TEST SET"
echo "=========================================="
if [ -f "results/llm_test.sql" ] && [ -f "records/llm_test.pkl" ]; then
    echo "⚠️  These are test predictions - cannot evaluate without ground truth"
    echo "    Files: results/llm_test.sql, records/llm_test.pkl"
else
    echo "⚠️  Files not found, skipping..."
fi

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo ""
echo "Note: Test set predictions cannot be evaluated locally."
echo "      Upload test files to Gradescope for evaluation."
