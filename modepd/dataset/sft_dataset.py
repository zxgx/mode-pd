import os
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
import random
from pathlib import Path

import torch
import datasets
from datasets import load_dataset, load_from_disk, DatasetDict


def build_sft_dataset(
    raw_dataset, 
    tokenizer, 
    block_size=4*1024, 
    logger=None,
    context_manager=None,
):
    # Ensure block_size is within tokenizer's limit
    if block_size is None:
        block_size = tokenizer.model_max_length
    else:
        if block_size > tokenizer.model_max_length and logger is not None:
            logger.warning(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)
    
    def tokenize_and_mask_function(examples):
        """
        Tokenize conversations and create loss masks for assistant responses only.
        
        This processes each conversation to ensure loss is only calculated on assistant messages.
        """
        # Process each conversation
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        
        # Iterate through each example
        for messages in examples["messages"]:
            conversation_tokens = []
            conversation_labels = []  # Track which positions should have loss calculated
            
            # Process all messages in a single pass
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                # Tokenize this message
                message_tokens = tokenizer.encode(
                    f"{role}: {content}",
                    add_special_tokens=False  # We'll add special tokens at the sequence level
                )
                
                # Set labels for this message - use actual token IDs for assistant, -100 for others
                if role == "assistant":
                    # For assistant messages, we'll compute loss - use the token IDs
                    message_labels = message_tokens.copy()
                else:
                    # For non-assistant messages, we mask the loss
                    message_labels = [-100] * len(message_tokens)
                
                # Add to the conversation
                conversation_tokens.extend(message_tokens)
                conversation_labels.extend(message_labels)
 
            # Truncate if too long           
            if tokenizer.bos_token_id is not None:
                sentence_delim_tokens = 2
            else:
                sentence_delim_tokens = 1
            if len(conversation_tokens) > block_size - sentence_delim_tokens:  # Account for special tokens
                conversation_tokens = conversation_tokens[:block_size - sentence_delim_tokens]
                conversation_labels = conversation_labels[:block_size - sentence_delim_tokens]

            if sum([each!=-100 for each in conversation_labels]) == 0:
                # all -100 labels lead to NaN loss
                continue

            if tokenizer.bos_token_id is not None:
                # Add special tokens
                input_ids = [tokenizer.bos_token_id] + conversation_tokens + [tokenizer.eos_token_id]
                # Add special token labels (-100 = no loss)
                labels = [-100] + conversation_labels + [-100]  # Don't compute loss on special tokens
            else:
                input_ids = conversation_tokens + [tokenizer.eos_token_id]
                labels = conversation_labels + [-100]  # Don't compute loss on special tokens
            attention_mask = [1] * len(input_ids)  # All tokens get attention

            # Append to batch lists
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels
        }
    
    with context_manager():
        # Process the dataset directly without concatenation
        return raw_dataset.map(
            tokenize_and_mask_function,
            batched=True,
            remove_columns=raw_dataset.column_names,
        )

def load_sft_dataset(
    dataset_name_or_path,
    tokenizer,
    split='train',
    block_size=4*1024,
    logger=None,
    accelerator=None,
    seed=None,
    cache_path=None,
):
    """
    Load and prepare a dataset for Supervised Fine-Tuning
    
    Args:
        tokenizer: The tokenizer to use
        streaming: Whether to stream the dataset
        split: Dataset split to load (typically "train")
        block_size: Maximum sequence length
        logger: Logger object 
        accelerator: Accelerator for distributed training
        seed: Random seed
        use_tulu: Whether to use the Tulu dataset (default) or another dataset
    
    Returns:
        Processed SFT dataset ready for training
    """
    if cache_path is not None and os.path.exists(cache_path):
        logger.info(f"Loading SFT dataset from cache: {cache_path})")
        return load_from_disk(cache_path)
    
    logger.info(f"Loading SFT dataset from: {dataset_name_or_path})")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    raw_dataset = load_dataset(dataset_name_or_path)

    # Create a context manager for distributed processing
    if accelerator is not None:
        context_manager = accelerator.main_process_first
    else:
        context_manager = nullcontext
    
    # Build the SFT dataset
    processed_dataset = build_sft_dataset(
        raw_dataset[split],
        tokenizer, 
        block_size=block_size,
        logger=logger,
        context_manager=context_manager,
    )
    processed_dataset = DatasetDict({split: processed_dataset})

    if cache_path is not None:
        with context_manager():
            processed_dataset.save_to_disk(cache_path)
    return processed_dataset


if __name__ == "__main__":
    import logging
    from transformers import AutoTokenizer

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    tokenizer_name_or_path = "/mnt/videodata/zhgeng/models/OLMoE-1B-7B-0125"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    dataset_name_or_path = "/mnt/videodata/zhgeng/datasets/tulu-3-sft-mixture"

    ds = load_sft_dataset(
        dataset_name_or_path, 
        tokenizer,
        block_size=4*1024,
        logger=logger,
        accelerator=None,
        seed=42,
        cache_path="/mnt/videodata/zhgeng/datasets/preprocessed-4k-tulu-3-sft-mixture")

    for i in range(4):
        print(ds['train'][i]['input_ids'][:5])
    