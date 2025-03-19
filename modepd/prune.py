import argparse

import torch
import torch.nn.functional
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling
)

from modepd.utils import register_custom_model, prepare_model_and_tokenizer, build_dataset, init_router
from modepd.pruning.layer_prune import layer_prune
from modepd.pruning.expert_prune import expert_prune
from modepd.pruning.weight_prune import weight_prune


register_custom_model()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-V2-Lite-Chat",)

    # parser.add_argument("--mod_type", type=str, default=None, choices=['staged', 'integrated'])
    # parser.add_argument("--staged_mod_topk", type=int, default=2048)

    # build a dataset for pruning
    parser.add_argument("--dataset_name_or_path", type=str, default="HuggingFaceFW/fineweb",) # "allenai/OLMoE-mix-0924"
    parser.add_argument("--dataset_config_name", type=str, default=None) # None
    parser.add_argument("--data_type", type=str, default=None)
    parser.add_argument("--streaming_dataset", action='store_true')

    parser.add_argument("--block_size", type=int, default=4*1024,)
    parser.add_argument("--max_steps", type=int, default=100,)

    # transformer layer pruning related arguments
    parser.add_argument("--layer_prune", action="store_true",)
    parser.add_argument("--layer_prune_metric", type=str, default='bi', choices=['cosine', 'angular'])
    parser.add_argument("--drop_n_layers", type=int, default=13,)

    # MoE expert pruning related arguments
    parser.add_argument("--expert_prune", action="store_true",)
    parser.add_argument("--expert_prune_metric", type=str, default='routing_score', choices=['routing_score', 'mc_smoe', 'ours'])
    parser.add_argument("--preserve_n_experts", type=int, default=30, help="Number of experts to preserve")
    parser.add_argument("--prune_using_fluctuation", action='store_true')
    
    # expert weight pruning related arguments
    parser.add_argument("--weight_prune", action="store_true",)
    parser.add_argument("--weight_prune_metric", type=str, default='norm', choices=["norm", "flap"])
    parser.add_argument("--preserve_channels_in_percent", type=float, default=0.7)

    parser.add_argument("--compressed_model_save_path", type=str, default="demo/DeepSeek-V2-Lite-Chat-Compressed",)
    return parser.parse_args()


def main():
    args = parse_args()

    model, tokenizer = prepare_model_and_tokenizer(args.model_name_or_path)
    model.eval()

    train_dataloader = None
    if not (args.weight_prune and args.weight_prune_metric == 'norm'):
        train_dataset = build_dataset(
            args.dataset_name_or_path, args.dataset_config_name, args.streaming_dataset, tokenizer, 'train', 
            args.data_type, args.block_size)
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
        new_model = model
            
    # Save
    if hasattr(new_model.config, "auto_map"):
        del new_model.config.auto_map
    if "auto_map" in tokenizer.init_kwargs:
        tokenizer.init_kwargs.pop("auto_map")
    new_model.save_pretrained(args.compressed_model_save_path)
    tokenizer.save_pretrained(args.compressed_model_save_path)


if __name__ == "__main__":
    # model.cuda()
    # text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
    # inputs = tokenizer(text, return_tensors="pt")
    # outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
    # result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(result)
    # exit(0)
    main()
    