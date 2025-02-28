from copy import deepcopy
import math
import logging

import torch
from torch import nn

from modepd.model.modeling_deepseek import DeepseekV2MoE, DeepseekV2PreTrainedModel, DeepseekV2ForCausalLM


@torch.no_grad()
def weight_prune_by_norm(args, model):
    # Move the model to the GPU 
    model.cuda()
    # Retrieves the index of the currently active GPU device
    device = torch.cuda.current_device()

    num_layer = model.config.num_hidden_layers
    routed_intermediate_indices, shared_intermediate_indices = {}, {}
    # new config content
    routed_intermediate_sizes, shared_intermediate_sizes = {}, {}

    for i in range(num_layer):
        mlp = model.model.layers[i].mlp
        if not isinstance(mlp, DeepseekV2MoE):
            continue
        
        # prune routed experts
        norm_list = []
        for expert in mlp.experts:
            norm_list.append(torch.norm(expert.down_proj.weight, dim=0))

        # n_exp * intermediate_size
        global_norm = torch.cat(norm_list)
        global_mask = torch.zeros_like(global_norm, dtype=torch.long, device=device)

        topk = int(math.ceil(global_norm.shape[0] * args.preserve_channels_in_percent))
        _, topk_indices = torch.topk(global_norm, topk)

        global_mask.scatter_(0, topk_indices, 1)
        expert_masks = torch.split(global_mask, model.config.moe_intermediate_size)
        routed_intermediate_indices[i] = []
        routed_intermediate_sizes[i] = []
        for ei, mask in enumerate(expert_masks):
            # new_intermediate_size
            cur_indices = torch.nonzero(mask).squeeze(1)
            new_intermediate_size = cur_indices.shape[0]
            if new_intermediate_size == 0:
                logging.warn(f"expert {ei} at layer {i} should have been fully pruned. for the sake of compatibility, we preserve 1 channel.")
                cur_norm = norm_list[ei]
                _, cur_indices = torch.topk(cur_norm, 1)
                new_intermediate_size = 1
            routed_intermediate_indices[i].append(cur_indices)
            routed_intermediate_sizes[i].append(new_intermediate_size)

        # prune shared experts
        shared_norm = torch.norm(mlp.shared_experts.down_proj.weight, dim=1)
        shared_mask = torch.zeros_like(shared_norm, dtype=torch.long, device=device)
        
        topk = int(math.ceil(shared_norm.shape[0] * args.preserve_channels_in_percent))
        _, topk_indices = torch.topk(shared_norm, topk)
        
        shared_intermediate_indices[i] = topk_indices
        shared_intermediate_sizes[i] = topk
    
    # update config
    new_config = deepcopy(model.config)
    new_config.routed_intermediate_sizes = routed_intermediate_sizes
    new_config.shared_intermediate_sizes = shared_intermediate_sizes

    # build new model
    new_model = DeepseekV2ForCausalLM(config=new_config)

    # prepare model state dict
    # model.cpu()
    state_dict = model.state_dict()
    for layer_idx in routed_intermediate_indices:
        for expert_idx, pruning_mask in enumerate(routed_intermediate_indices[layer_idx]):
            # pruning_mask = pruning_mask.cpu()
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"][pruning_mask]
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"][pruning_mask]
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"][:, pruning_mask]

        shared_mask = shared_intermediate_indices[layer_idx] #.cpu()
        state_dict[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"][shared_mask]
        state_dict[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"][shared_mask]
        state_dict[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"][:, shared_mask]

    new_model.load_state_dict(state_dict, strict=True)
    if not hasattr(new_model, "quantization_config"):
        new_model.bfloat16()

    return new_model


def weight_prune(args, model, train_dataloader):
    if args.weight_prune_metric == 'norm':
        return weight_prune_by_norm(args, model)


