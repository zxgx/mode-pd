import argparse
from copy import deepcopy

import torch
import torch.nn.functional
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    DataCollatorForLanguageModeling
)

from modepd.utils import build_dataset
from modepd.model.modeling_deepseek import DeepseekV2PreTrainedModel, DeepseekV2ForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-V2-Lite-Chat",)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb",)
    parser.add_argument("--dataset_config_name", type=str, default="sample-350BT",)
    parser.add_argument("--block_size", type=int, default=4*1024,)
    parser.add_argument("--max_steps", type=int, default=100,)
    parser.add_argument("--drop_n", type=int, default=1,)
    parser.add_argument("--compressed_model_save_path", type=str, default="deepseek-ai/DeepSeek-V2-Lite-Chat-Compressed",)

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
    similarities = torch.zeros(num_layers, num_layers, device=device)
    denominator = torch.zeros(num_layers, num_layers, device=device)
    cache, handles = [None for _ in range(num_layers+1)], []

    # register hooks which save the input for each layer and the output of the last layer
    for i in range(num_layers):
        layer = model.model.layers[i]
        
        # bind layer index i into this hook to record input/outputs
        def stateful_hook(module, _input, _output):
            cache[i] = _input.squeeze(0)
            if i == num_layers-1:
                cache[-1] = _output.squeeze(0)

                # accumulate results 
                for j in range(num_layers):
                    for k in range(j+1, num_layers+1):
                        sim = torch.nn.functional.cosine_similarity(cache[j].float(), cache[k].float(), dim=0)
                        denominator[j, k-j-1] += sim.shape[0]
                        similarities[j, k-j-1] += sim.sum()
                    
        handle = layer.register_forward_hook(stateful_hook)
        handles.append(handle)
    
    # execute model to collect similarities
    for step, batch in enumerate(train_dataloader):
        if step == args.max_steps:
            break
        with torch.no_grad():
            model(**batch)
    
    similarities /= denominator

    # clear handles before saving
    for handle in handles:
        handle.remove()

    # prune the model
    similarities_drop_1 = similarities[:, 0].view(-1)
    sorted_similarities, sorted_layer_id = torch.sort(similarities_drop_1, dim=0, descending=True)
    dropped_layer_list = sorted_layer_id[:args.drop_n].tolist()

    reserved_layer_list = sorted(list(set(range(num_layers)) - set(dropped_layer_list)))
    layer_id_mapping = {}
    for new_id, reserved_old_id in enumerate(reserved_layer_list):
        layer_id_mapping[reserved_old_id] = new_id

    state_dict = model.state_dict()
    save_state_dict = {}
    for state_name in sorted(list(state_dict.keys())):
        for old_layer_id, new_layer_id in layer_id_mapping.items():
            if f"layers.{old_layer_id}." in state_name:  # convert old ids to new ones
                save_state_dict[state_name.replace(f"layers.{old_layer_id}", f"layers.{new_layer_id}")] = state_dict[state_name]
                break
            elif f"layers." not in state_name:  # copy other states
                save_state_dict[state_name] = state_dict[state_name]
                break

    # Config
    new_config = deepcopy(model.config)
    new_config.num_hidden_layers = len(layer_id_mapping)

    preserved_layers = sorted([int(s) for s in layer_id_mapping.keys()])

    if isinstance(model, DeepseekV2PreTrainedModel):
        if hasattr(new_config, "layer_experts_idx"):  # for compatibility with Expert Drop
            new_config.layer_experts_idx = [model.config.layer_experts_idx[i] for i in preserved_layers]
        if isinstance(new_config.n_routed_experts, list):  # for compatibility with Expert Drop & Layer Drop
            new_config.n_routed_experts = [model.config.n_routed_experts[i] for i in preserved_layers]
        new_model = DeepseekV2ForCausalLM(config=new_config)
    else:
        raise NotImplementedError

    # Model
    new_model.load_state_dict(save_state_dict, strict=True)  # update the layer parameters
    if not hasattr(new_model, "quantization_config"):
        new_model.bfloat16()

    # Save
    new_model.save_pretrained(args.compressed_model_save_path)
    tokenizer.save_pretrained(args.compressed_model_save_path)
    new_config.save_pretrained(args.compressed_model_save_path)



if __name__ == "__main__":
    main()
    