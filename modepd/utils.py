import os
from contextlib import nullcontext
from itertools import chain
import math

import torch
from datasets import (
    load_dataset, 
    # interleave_datasets,
)
from transformers import AutoConfig, GenerationConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM

from modepd.model.deepseek_v2.configuration_deepseek import DeepseekV2Config
from modepd.model.deepseek_v2.modeling_deepseek import DeepseekV2Model, DeepseekV2ForCausalLM
from modepd.model.moonshotai.configuration_deepseek import DeepseekV3Config
from modepd.model.moonshotai.modeling_deepseek import DeepseekV3Model, DeepseekV3ForCausalLM
from modepd.model.moonshotai.tokenization_moonshot import TikTokenTokenizer


GB = 1024**3


def register_custom_model():
    AutoConfig.register("deepseek_v2_compressed", DeepseekV2Config)
    AutoModel.register(DeepseekV2Config, DeepseekV2Model)
    AutoModelForCausalLM.register(DeepseekV2Config, DeepseekV2ForCausalLM)

    AutoConfig.register("deepseek_v3_compressed", DeepseekV3Config)
    AutoModel.register(DeepseekV3Config, DeepseekV3Model)
    AutoModelForCausalLM.register(DeepseekV3Config, DeepseekV3ForCausalLM)
    AutoTokenizer.register(DeepseekV3Config, TikTokenTokenizer)


def prepare_model_and_tokenizer(model_name_or_path, mode='prune', use_cache=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    causal_model_class = AutoModelForCausalLM
    if "DeepSeek-V2" in model_name_or_path and mode == 'prune':
        causal_model_class = DeepseekV2ForCausalLM
    elif "Moonlight-16B-A3B" in model_name_or_path and mode == 'prune':
        causal_model_class = DeepseekV3ForCausalLM
    else:
        raise KeyError(f"unsupport model: {model_name_or_path}")
    model = causal_model_class.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, use_cache=use_cache, attn_implementation="flash_attention_2",
    )

    if "DeepSeek-V2" in model_name_or_path:
        model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return model, tokenizer


def get_memory_stats():
    alloc = torch.cuda.memory_allocated() / GB
    max_alloc = torch.cuda.max_memory_allocated() / GB
    reserved = torch.cuda.memory_reserved() / GB
    max_reserved = torch.cuda.max_memory_reserved() / GB
    return alloc, max_alloc, reserved, max_reserved


def build_dataset(
    dataset_name_or_path, dataset_config_name, streaming, tokenizer, split, 
    data_type=None, block_size=4*1024, logger=None, accelerator=None,
):
    main_process_context = accelerator.main_process_first if accelerator is not None else nullcontext

    #################
    # Prepare dataset
    # TODO: mix fineweb (en) and fineweb-2 (zh) dataset
    # en = load_dataset("HuggingFaceFW/fineweb", name="sample-350BT", split="train", streaming=True)
    # zh = load_dataset("HuggingFaceFW/fineweb-2", name="cmn_Hani", split="train", streaming=True)
    # ds = interleave_datasets([en, zh], probabilities=[0.8, 0.2], seed=42)
    # ds = en
    if os.path.exists(dataset_name_or_path) and data_type is not None:
        raw_dataset = load_dataset(data_type, data_dir=dataset_name_or_path, name=dataset_config_name, split=split, streaming=streaming)
    else:
        raw_dataset = load_dataset(dataset_name_or_path, dataset_config_name, split=split, streaming=streaming)
    
    # raw_iter = iter(raw_datasets['train'])
    # for index in range(8):
    #     sample = next(raw_iter)
    #     logger.info(f"rank: {accelerator.process_index}/{accelerator.num_processes} raw sample {index}: {sample['text']}", main_process_only=False)
    
    #################
    # Preprocessing the datasets.
    # 1. Only load text fields for the dataloader
    column_names = raw_dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with main_process_context():
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

    # 2. Padding to max length    
    if block_size is None:
        block_size = tokenizer.model_max_length
    else:
        if block_size > tokenizer.model_max_length and logger is not None:
            logger.warning(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)

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
        lm_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
        )

    # fine-web doesn't have validation set
    # eval_dataset = lm_datasets["validation"]
    # this logic will be handled by `accelerator.prepare`
    # if accelerator.num_processes > 1:
    #     train_dataset = split_dataset_by_node(train_dataset, rank=accelerator.process_index, world_size=accelerator.num_processes)

    # raw_iter = iter(dataset)
    # for index in range(4):
    #     sample = next(raw_iter)
    #     logger.info(f"rank: {accelerator.process_index}/{accelerator.num_processes} main sample {index}: {tokenizer.decode(sample['input_ids'])}", main_process_only=False)
    
    return lm_dataset


def init_router(model, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    if model.config.mod_type == 'integrated':
        for layer in model.model.layers:
            if hasattr(layer.mlp, "gate") and layer.mlp.gate.skip_router_weight is not None:
                torch.nn.init.kaiming_uniform_(layer.mlp.gate.skip_router_weight, a=math.sqrt(5))
    elif model.config.mod_type == 'staged':
        for layer in model.model.layers:
            if hasattr(layer, "mod_router") and layer.mod_router is not None:
                torch.nn.init.kaiming_uniform_(layer.mod_router.token_router.weight)

