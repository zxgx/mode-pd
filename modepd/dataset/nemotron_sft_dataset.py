import json
import logging
from itertools import chain
import torch

from datasets import load_dataset, interleave_datasets

logger = logging.getLogger(__name__)

def _format_and_tokenize(example, tokenizer):
    """
    Formats a single SFT example using the tokenizer's chat template,
    tokenizes it, and creates labels with prompt masking.
    """
    # 1. Parse the input into a list of messages (the prompt)
    try:
        if isinstance(example['input'], str) and example['input']:
            prompt_messages = json.loads(example['input'])
        else:
            prompt_messages = example.get('input') or []
    except (json.JSONDecodeError, TypeError):
        prompt_messages = [{"role": "user", "content": str(example.get('input'))}]

    # 2. Get the assistant's response
    response_str = example.get('output', '') or ""
    if not response_str: # Skip examples with no response
        return None, None

    # 3. Apply the chat template to the prompt part.
    # We add the generation prompt to get the tokens that lead up to the assistant's response.
    prompt_tokens = tokenizer.apply_chat_template(
        prompt_messages, 
        add_generation_prompt=True, 
        tokenize=True, 
        add_special_tokens=False
    )

    # 4. Apply the chat template to the full conversation.
    full_messages = prompt_messages + [{"role": "assistant", "content": response_str}]
    full_tokens = tokenizer.apply_chat_template(
        full_messages, 
        tokenize=True, 
        add_special_tokens=False
    )
    
    # Add EOS token at the end of the full sequence. This is standard for SFT.
    if tokenizer.eos_token_id is not None:
        full_tokens.append(tokenizer.eos_token_id)

    # 5. Create labels by masking the prompt part.
    labels = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]

    # Final input_ids are the full token sequence.
    input_ids = full_tokens
    
    if len(input_ids) != len(labels):
        logger.warning("Mismatch between input_ids and labels length after applying chat template. Skipping example.")
        return None, None
        
    return input_ids, labels

def build_packed_sft_dataset(
    dataset_name, config_name, tokenizer, streaming=True, seed=None, 
    split='train', block_size=32768, num_validation_samples=None
):
    """
    Builds a dataset for SFT with sequence packing.

    Args:
        dataset_name (str or list): Name of the dataset or list of splits.
        tokenizer: The tokenizer to use.
        block_size (int): The desired sequence length for packing.
        ... and other dataset args.

    Returns:
        An iterable dataset yielding packed sequences.
    """
    logger.info(f"Building packed SFT dataset for split '{split}' with block size {block_size}")

    # Load the raw dataset, handling multiple splits if necessary
    if isinstance(split, list):
        datasets_to_interleave = [
            load_dataset(dataset_name, name=config_name, split=s, streaming=streaming) for s in split
        ]
        raw_dataset = interleave_datasets(datasets_to_interleave, seed=seed)
    else:
        raw_dataset = load_dataset(dataset_name, name=config_name, split=split, streaming=streaming)

    # For creating a validation set from the training data
    if num_validation_samples:
        raw_dataset = raw_dataset.shuffle(seed=seed, buffer_size=10_000).take(num_validation_samples)
    
    def packed_sequence_generator():
        buffer_input_ids = []
        buffer_labels = []

        for example in raw_dataset:
            input_ids, labels = _format_and_tokenize(example, tokenizer)
            if input_ids is None:
                continue
            
            buffer_input_ids.extend(input_ids)
            buffer_labels.extend(labels)

            while len(buffer_input_ids) >= block_size:
                packed_input_ids = buffer_input_ids[:block_size]
                packed_labels = buffer_labels[:block_size]

                yield {
                    "input_ids": torch.tensor(packed_input_ids, dtype=torch.long),
                    "labels": torch.tensor(packed_labels, dtype=torch.long),
                    "attention_mask": torch.ones(block_size, dtype=torch.long)
                }
                
                # Carry over the remainder to the next iteration
                buffer_input_ids = buffer_input_ids[block_size:]
                buffer_labels = buffer_labels[block_size:]

    return packed_sequence_generator() 