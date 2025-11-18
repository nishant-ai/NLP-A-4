import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle
import json

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt', quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Optimized dataset class for T5 with better prompt engineering
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split
        self.process_data(data_folder, split, self.tokenizer)
        print(f"Dataset Initialized: {split} with {len(self)} examples")

    def process_data(self, data_folder, split, tokenizer):
        # 1. Load the Dataset
        nl_file = os.path.join(data_folder, f'{split}.nl')
        nl_queries = load_lines(nl_file)

        # 2. T5 task prefix format: Simple "translate to SQL:" prefix
        # No schema context needed - model learns from training data
        prompts = []
        for query in nl_queries:
            # Clean T5 task format: "task_prefix: input_text"
            prompt = f"translate to SQL: {query}"
            prompts.append(prompt)

        # 3. Tokenize inputs with proper max_length
        self.encoder_inputs = []
        for p in prompts:
            tokens = tokenizer.encode(
                p,
                truncation=True,
                max_length=128,  # Reduced since we removed schema context
                add_special_tokens=True
            )
            self.encoder_inputs.append(tokens)

        if split != 'test':
            # Load SQL targets
            sql_file = os.path.join(data_folder, f'{split}.sql')
            sql_queries = load_lines(sql_file)

            # 4. Tokenize targets with appropriate max_length
            self.decoder_targets = []
            for sql in sql_queries:
                tokens = tokenizer.encode(
                    sql, 
                    truncation=True, 
                    max_length=256,  # SQL queries are typically shorter
                    add_special_tokens=True
                )
                self.decoder_targets.append(tokens)
        else:
            self.decoder_targets = None

        # Store decoder start token (pad_token_id for T5)
        self.bos_token_id = tokenizer.pad_token_id
    
    def __len__(self):
        '''Return the number of examples in the dataset'''
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        if self.split != 'test':
            # For train/dev: return encoder input, decoder input, decoder target
            encoder_input = torch.tensor(self.encoder_inputs[idx], dtype=torch.long)
            decoder_target = torch.tensor(self.decoder_targets[idx], dtype=torch.long)
            
            # Decoder input: prepend BOS token to target
            decoder_input = torch.cat([
                torch.tensor([self.bos_token_id], dtype=torch.long),
                decoder_target
            ])
            
            return encoder_input, decoder_input, decoder_target
        else:
            # For test: return encoder input and initial decoder token
            encoder_input = torch.tensor(self.encoder_inputs[idx], dtype=torch.long)
            initial_token = torch.tensor([self.bos_token_id], dtype=torch.long)
            
            return encoder_input, initial_token

def normal_collate_fn(batch):
    '''
    Optimized collation function with proper padding values
    '''
    # Unpack batch
    encoder_inputs = [item[0] for item in batch]
    decoder_inputs = [item[1] for item in batch]
    decoder_targets = [item[2] for item in batch]

    # Pad sequences
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_input_ids = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    # OPTIMIZED: Use -100 for padding in targets (ignored by loss function)
    decoder_target_ids = pad_sequence(decoder_targets, batch_first=True, padding_value=-100)

    # Create attention mask for encoder (1 for real tokens, 0 for padding)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # Get initial decoder inputs
    initial_decoder_inputs = decoder_input_ids[:, 0]

    return encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function for test set inference
    '''
    # Unpack batch
    encoder_inputs = [item[0] for item in batch]
    initial_tokens = [item[1] for item in batch]

    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)

    # Create attention mask for encoder
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # Stack initial decoder tokens
    initial_decoder_inputs = torch.stack(initial_tokens).squeeze()
    if initial_decoder_inputs.dim() == 0:  # Handle single item batch
        initial_decoder_inputs = initial_decoder_inputs.unsqueeze(0)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    # OPTIMIZED: Add num_workers for faster data loading
    dataloader = DataLoader(
        dset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        num_workers=2,  # Parallel data loading
        pin_memory=torch.cuda.is_available()  # Faster GPU transfer
    )
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader

def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_schema(schema_path):
    '''OPTIMIZED: Create more concise schema representation'''
    with open(schema_path, 'r') as f:
        schema_data = json.load(f)

    # Create simplified schema focusing on table and column names
    tables = []
    for table_name, columns in schema_data['ents'].items():
        # Only include column names, not types (to save tokens)
        col_names = list(columns.keys())
        # Limit columns if there are too many
        if len(col_names) > 10:
            col_names = col_names[:10] + ['...']
        tables.append(f"{table_name}({','.join(col_names)})")
    
    # Join tables with separator
    schema_str = " | ".join(tables)
    
    # Truncate if still too long (keep most important tables)
    if len(schema_str) > 1000:
        schema_str = schema_str[:1000] + "..."
    
    return schema_str

def load_prompting_data(data_folder):
    # TODO: Implement if needed for prompting experiments
    pass

if __name__ == "__main__":
    print("=" * 80)
    print("Testing Optimized T5Dataset")
    print("=" * 80)

    data_folder = 'data'

    # Test train split
    print("\nLoading TRAIN dataset")
    dataset = T5Dataset(data_folder, 'train')
    print(f"Dataset size: {len(dataset)}")
    
    # Show first example
    encoder_input, decoder_input, decoder_target = dataset[0]
    print(f"\nFirst example shapes:")
    print(f"  Encoder input: {encoder_input.shape}")
    print(f"  Decoder input: {decoder_input.shape}")
    print(f"  Decoder target: {decoder_target.shape}")
    
    # Decode to see the prompt format
    print(f"\nPrompt (first 300 chars):")
    prompt = dataset.tokenizer.decode(encoder_input)
    print(f"  {prompt[:300]}...")
    
    print(f"\nTarget SQL:")
    target = dataset.tokenizer.decode(decoder_target, skip_special_tokens=True)
    print(f"  {target}")
    
    # Test dataloader
    print("\n" + "=" * 80)
    print("Testing DataLoader")
    print("=" * 80)
    
    train_loader, dev_loader, test_loader = load_t5_data(batch_size=4, test_batch_size=8)
    
    for batch in train_loader:
        encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_inputs = batch
        print(f"Batch shapes:")
        print(f"  Encoder IDs: {encoder_ids.shape}")
        print(f"  Encoder mask: {encoder_mask.shape}")
        print(f"  Decoder inputs: {decoder_input_ids.shape}")
        print(f"  Decoder targets: {decoder_target_ids.shape}")
        print(f"  Initial decoder: {initial_decoder_inputs.shape}")
        
        # Check for -100 in decoder targets (padding)
        print(f"\nDecoder targets contain -100 padding: {(-100 in decoder_target_ids).item()}")
        break