import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    DataCollatorForLanguageModeling
)

from modepd.utils import build_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-V2-Lite-Chat",)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb",)
    parser.add_argument("--dataset_config_name", type=str, default="sample-350BT",)
    parser.add_argument("--block_size", type=int, default=4*1024,)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.cuda.current_device()
    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, use_cache=False)
    
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
    )
    for batch in train_dataloader:
        print(f"{batch['input_ids'].shape}")
        break

    num_layers = model.config.num_hidden_layers
    similarities = torch.full(num_layers, num_layers, -float('inf'), device=device)
    cache, handles = [], []

    # register hooks which save the input for each layer and the output of the last layer
    for i in range(num_layers):
        layer = model.model.layers[i]
        
        def stateful_hook(module, _input, _output):
            cache.append([])

            cache[i].append(_input.to('cpu', non_blocking=True))

            if i == num_layers-1:
                cache.append([])
                cache[i+1].append(_output.to('cpu', non_blocking=True))

        handle = layer.register_forward_hook(stateful_hook)
        handles.append(handle)
    
    # clear handles before saving
    for handle in handles:
        handle.remove()

        

if __name__ == "__main__":
    main()
    