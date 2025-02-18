import argparse

import torch
import torch.nn.functional
from torch.utils.data import DataLoader
from transformers import (
    # AutoTokenizer, AutoModelForCausalLM, 
    GenerationConfig,
    DataCollatorForLanguageModeling
)

from modepd.utils import build_dataset
from modepd.model.modeling_deepseek import DeepseekV2ForCausalLM
from modepd.model.tokenization_deepseek_fast import DeepseekTokenizerFast
from modepd.pruning.layer_prune import layer_prune
from modepd.pruning.expert_prune import expert_prune
from modepd.pruning.weight_prune import weight_prune


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-V2-Lite-Chat",)
    parser.add_argument("--enable_mod", action="store_true",)
    
    # build a dataset for pruning
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb",)
    parser.add_argument("--dataset_config_name", type=str, default="sample-350BT",)
    parser.add_argument("--block_size", type=int, default=4*1024,)
    parser.add_argument("--max_steps", type=int, default=100,)

    # transformer layer pruning related arguments
    parser.add_argument("--layer_prune", action="store_true",)
    parser.add_argument("--drop_n", type=int, default=13,)
    parser.add_argument("--compressed_model_save_path", type=str, default="demo/DeepSeek-V2-Lite-Chat-Compressed",)

    # MoE expert pruning related arguments
    parser.add_argument("--expert_prune", action="store_true",)

    # weight pruning related arguments
    parser.add_argument("--weight_prune", action="store_true",)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name_or_path
    tokenizer = DeepseekTokenizerFast.from_pretrained(model_name)
    model = DeepseekV2ForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, use_cache=False, attn_implementation="flash_attention_2",
        enable_mod=args.enable_mod,
    )
    
    if "DeepSeek-V2" in model_name:
        model.generation_config = GenerationConfig.from_pretrained(model_name)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    model.eval()

    import pdb
    pdb.set_trace()

    train_dataset = build_dataset(args, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    )

    for batch in train_dataloader:
        print(f"{batch['input_ids'].shape}")
        break

    if args.layer_prune:
        new_model = layer_prune(args, model, train_dataloader)
    elif args.expert_prune:
        new_model = expert_prune(args, model, train_dataloader)
    elif args.weight_prune:
        new_model = weight_prune(args, model, train_dataloader)
    else:
        raise NotImplementedError("Only layer prune is supported for now")
            
    # Save
    new_model.save_pretrained(args.compressed_model_save_path)
    tokenizer.save_pretrained(args.compressed_model_save_path)


if __name__ == "__main__":
    main()
    