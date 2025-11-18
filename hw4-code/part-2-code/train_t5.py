import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training - keeping original structure
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=3e-4)  # Optimized
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--scheduler_type', type=str, default="linear", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=2,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=20,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=5,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='optimized',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    
    experiment_name = 'ft_experiment'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb and wandb is not None:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if args.patience_epochs > 0 and epochs_since_improvement >= args.patience_epochs:
            print(f"Early stopping triggered. Best F1: {best_f1}")
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    '''
    Optimized training epoch with proper loss computation
    '''
    model.train()
    total_loss = 0
    total_steps = 0

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)
        
        # CRITICAL FIX: Use labels directly without manual shifting
        # T5 handles the shifting internally when labels are provided
        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=decoder_targets  # Pass full target sequence as labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        total_steps += 1
    
    return total_loss / total_steps

def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Optimized evaluation with beam search for better generation
    '''
    model.eval()
    total_loss = 0
    total_steps = 0
    
    # Get tokenizer for decoding
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    generated_queries = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            
            # Compute loss
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                labels=decoder_targets
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1
            
            # Generate SQL queries with beam search for better quality
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=256,
                num_beams=10,  # OPTIMIZED: Use beam search instead of greedy
                num_return_sequences=1,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Deterministic generation
                temperature=1.0,
                length_penalty=1.0  # Neutral length penalty
            )
            
            # Decode generated queries
            batch_queries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            generated_queries.extend(batch_queries)
    
    # Save generated queries and compute records
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    
    # Compute metrics
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    
    # Compute error rate
    num_errors = sum(1 for msg in error_msgs if msg != "")
    error_rate = num_errors / len(error_msgs) if len(error_msgs) > 0 else 0
    
    eval_loss = total_loss / total_steps if total_steps > 0 else 0
    
    return eval_loss, record_f1, record_em, sql_em, error_rate

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Optimized test inference with beam search
    '''
    model.eval()
    
    # Get tokenizer for decoding
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    generated_queries = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate SQL queries with beam search
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=256,
                num_beams=10,  # OPTIMIZED: Beam search
                num_return_sequences=1,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,
                length_penalty=1.0
            )
            
            # Decode generated queries
            batch_queries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            generated_queries.extend(batch_queries)
    
    # Save generated queries and compute records
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    print(f"Test inference complete. Saved {len(generated_queries)} queries to {model_sql_path}")

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)
    
    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
    
    # Train
    train(args, model, train_loader, dev_loader, optimizer, scheduler)
    
    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = 'ft_experiment'
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")
    
    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()