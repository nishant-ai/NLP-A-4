#!/usr/bin/env python
"""
Generate test predictions from a saved checkpoint for extra credit submission.
Usage: python generate_test_predictions.py --checkpoint_path <path_to_best_model.pt>
"""

import os
import argparse
from tqdm import tqdm
import torch
from transformers import T5TokenizerFast

from t5_utils import initialize_model, load_model_from_checkpoint
from load_data import load_t5_data
from utils import save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_args():
    parser = argparse.ArgumentParser(description='Generate test predictions from checkpoint')

    # Model checkpoint
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint (e.g., checkpoints/scr_experiments/scr_experiment/best_model.pt)')
    parser.add_argument('--finetune', action='store_true',
                        help='Set this if the checkpoint is from a finetuned model (not from scratch)')

    # Data hyperparameters (must match training)
    parser.add_argument('--test_batch_size', type=int, default=16)

    # Output paths
    parser.add_argument('--output_sql', type=str, default='results/t5_scr_experiment_ec_test.sql',
                        help='Path to save generated SQL queries')
    parser.add_argument('--output_records', type=str, default='records/t5_scr_experiment_ec_test.pkl',
                        help='Path to save generated database records')

    return parser.parse_args()

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Run inference on test set to generate SQL queries and database records.
    '''
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    all_generated_queries = []

    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader, desc="Test inference"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            # Generate SQL queries
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=1,  # Greedy decoding
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode generated queries
            generated_queries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_generated_queries.extend(generated_queries)

    # Save generated queries and records
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(all_generated_queries, model_sql_path, model_record_path)

    print(f"Generated {len(all_generated_queries)} SQL queries for test set")
    print(f"Saved SQL to: {model_sql_path}")
    print(f"Saved records to: {model_record_path}")

def main():
    print("=" * 50)
    print("GENERATE TEST PREDICTIONS - Extra Credit")
    print("=" * 50)

    args = get_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Checkpoint not found at {args.checkpoint_path}")
        print("\nCommon locations:")
        print("  - checkpoints/scr_experiments/scr_experiment/best_model.pt")
        print("  - checkpoints/ft_experiments/ft_experiment/best_model.pt")
        return

    print(f"Loading checkpoint from: {args.checkpoint_path}")

    # Load test data
    print("Loading test data...")
    _, _, test_loader = load_t5_data(batch_size=16, test_batch_size=args.test_batch_size)

    # Initialize and load model
    print("Initializing model...")
    model = initialize_model(args)

    print("Loading model weights from checkpoint...")
    checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    print(f"Model loaded successfully on device: {DEVICE}")

    # Run inference
    print("\nRunning inference on test set...")
    test_inference(args, model, test_loader, args.output_sql, args.output_records)

    print("\n" + "=" * 50)
    print("DONE! Submission files created:")
    print(f"  - {args.output_sql}")
    print(f"  - {args.output_records}")
    print("=" * 50)

if __name__ == "__main__":
    main()
