from contextlib import nullcontext
from itertools import chain

import torch
from datasets import (
    load_dataset, 
    # interleave_datasets,
)


GB = 1024**3


def get_memory_stats():
    alloc = torch.cuda.memory_allocated() / GB
    max_alloc = torch.cuda.max_memory_allocated() / GB
    reserved = torch.cuda.memory_reserved() / GB
    max_reserved = torch.cuda.max_memory_reserved() / GB
    return alloc, max_alloc, reserved, max_reserved


def build_dataset(args, tokenizer, logger=None, accelerator=None):
    main_process_context = accelerator.main_process_first if accelerator is not None else nullcontext

    #################
    # Prepare dataset
    # TODO: mix fineweb (en) and fineweb-2 (zh) dataset
    # en = load_dataset("HuggingFaceFW/fineweb", name="sample-350BT", split="train", streaming=True)
    # zh = load_dataset("HuggingFaceFW/fineweb-2", name="cmn_Hani", split="train", streaming=True)
    # ds = interleave_datasets([en, zh], probabilities=[0.8, 0.2], seed=42)
    # ds = en
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, streaming=True)
    
    #################
    # Preprocessing the datasets.
    # 1. Only load text fields for the dataloader
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with main_process_context():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

    # 2. Padding to max length    
    if args.block_size is None:
        block_size = tokenizer.model_max_length
    else:
        if args.block_size > tokenizer.model_max_length and logger is not None:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with main_process_context():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )
    
    train_dataset = lm_datasets["train"]
    # fine-web doesn't have validation set
    # eval_dataset = lm_datasets["validation"]
    # this logic will be handled by `accelerator.prepare`
    # if accelerator.num_processes > 1:
    #     train_dataset = split_dataset_by_node(train_dataset, rank=accelerator.process_index, world_size=accelerator.num_processes)

    return train_dataset
