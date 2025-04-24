from copy import deepcopy
import math
import logging
from tqdm import tqdm

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from modepd.model.deepseek_v2.modeling_deepseek import DeepseekV2MoE, DeepseekV2MLP
from modepd.model.moonshotai.modeling_deepseek import DeepseekV3MoE, DeepseekV3MLP
from modepd.model.olmoe.modeling_olmoe import OlmoeSparseMoeBlock


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
        if not isinstance(mlp, (DeepseekV2MoE, DeepseekV3MoE, OlmoeSparseMoeBlock)):
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
    new_model = AutoModelForCausalLM.from_config(config=new_config)

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


@torch.no_grad()
def weight_prune_by_flap(args, model, train_dataloader):
    # Move the model to the GPU 
    model.cuda()
    # Retrieves the index of the currently active GPU device
    device = torch.cuda.current_device()

    handles = []
    num_layer = model.config.num_hidden_layers
    routed_intermediate_indices, shared_intermediate_indices = {}, {}
    # new config content
    routed_intermediate_sizes, shared_intermediate_sizes = {}, {}

    bias_stats = {}
    def create_hook(linear_name):
        def stateful_hook(module, _input, _output):
            inp = _input[0]
            batch_size = 1
            inp = inp.view(-1, inp.shape[-1]).t()

            if inp.shape[1] == 0:
                return

            # retrieve stats
            num_samples = bias_stats[linear_name]["num_samples"]
            baseline_inp = bias_stats[linear_name]["baseline_inp"]
            fluc_inp = bias_stats[linear_name]["fluc_inp"]
            
            # update moving average and fluctuation
            baseline_inp *= num_samples / (num_samples + batch_size)
            baseline_inp += torch.mean(inp, dim=1) / (num_samples + batch_size)

            if num_samples > 0:
                fluc_inp *= (num_samples - 1) / (num_samples + batch_size - 1)
                fluc_inp += torch.sum((inp - baseline_inp.unsqueeze(1))**2, dim=1) / (num_samples + batch_size)

            # write back stats
            bias_stats[linear_name]["num_samples"] = num_samples + batch_size
            bias_stats[linear_name]["baseline_inp"] = baseline_inp
            bias_stats[linear_name]["fluc_inp"] = fluc_inp

        return stateful_hook

    for i in range(num_layer):
        mlp = model.model.layers[i].mlp
        if isinstance(mlp, (DeepseekV2MLP, DeepseekV3MLP)):
            inp_dim_size = mlp.down_proj.weight.shape[1]
            module_name = f"layers.{i}.mlp"
            bias_stats[module_name] = {
                "baseline_inp": torch.zeros(inp_dim_size, device=device),
                "fluc_inp": torch.zeros(inp_dim_size, device=device),
                "num_samples": 0
            }
            
            handle = mlp.down_proj.register_forward_hook(create_hook(module_name))
            handles.append(handle)
        elif isinstance(mlp, (DeepseekV2MoE, DeepseekV3MoE, OlmoeSparseMoeBlock)):
            num_experts = len(mlp.experts)
            for e_i in range(num_experts):
                inp_dim_size = mlp.experts[e_i].down_proj.weight.shape[1]
                module_name = f"layers.{i}.mlp.experts.{e_i}"
                bias_stats[module_name] = {
                    "baseline_inp": torch.zeros(inp_dim_size, device=device),
                    "fluc_inp": torch.zeros(inp_dim_size, device=device),
                    "num_samples": 0
                }
                handle = mlp.experts[e_i].down_proj.register_forward_hook(create_hook(module_name))
                handles.append(handle)

            if isinstance(mlp, (DeepseekV2MoE, DeepseekV3MoE)):
                shared_inp_dim_size = mlp.shared_experts.down_proj.weight.shape[1]
                shared_module_name = f"layers.{i}.mlp.shared_experts"
                bias_stats[shared_module_name] = {
                    "baseline_inp": torch.zeros(shared_inp_dim_size, device=device),
                    "fluc_inp": torch.zeros(shared_inp_dim_size, device=device),
                    "num_samples": 0
                }

                handle = mlp.shared_experts.down_proj.register_forward_hook(create_hook(shared_module_name))
                handles.append(handle)
        else:
            raise ValueError(f"unknow mlp type: {type(mlp)} at layer {i}")
    
    data_iter = iter(train_dataloader)
    for step in tqdm(range(args.max_steps), desc="collecting accumulated stats"):
        batch = next(data_iter)
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    dense_fluc_metric_list, dense_baseline_list = [], []
    expert_fluc_metric_list, expert_baseline_list = [], []
    shared_fluc_metric_list, shared_baseline_list = [], []
    for i in range(num_layer):
        mlp = model.model.layers[i].mlp
        if isinstance(mlp, (DeepseekV2MLP, DeepseekV3MLP)):
            module_name = f"layers.{i}.mlp"
            dense_fluc_metric_list.append(
                bias_stats[module_name]["fluc_inp"] * torch.sum(mlp.down_proj.weight.data.pow(2), dim=0)
            )
            dense_baseline_list.append(
                bias_stats[module_name]["baseline_inp"]
            )
        elif isinstance(mlp, (DeepseekV2MoE, DeepseekV3MoE, OlmoeSparseMoeBlock)):
            num_experts = len(mlp.experts)
            for e_i in range(num_experts):
                module_name = f"layers.{i}.mlp.experts.{e_i}"
                expert_fluc_metric_list.append(
                    bias_stats[module_name]["fluc_inp"] * torch.sum(mlp.experts[e_i].down_proj.weight.data.pow(2), dim=0)
                )
                expert_baseline_list.append(
                    bias_stats[module_name]["baseline_inp"]
                )
            if isinstance(mlp, (DeepseekV2MoE, DeepseekV3MoE)):
                shared_module_name = f"layers.{i}.mlp.shared_experts"
                shared_fluc_metric_list.append(
                    bias_stats[shared_module_name]["fluc_inp"] * torch.sum(mlp.shared_experts.down_proj.weight.data.pow(2), dim=0)
                )
                shared_baseline_list.append(
                    bias_stats[shared_module_name]["baseline_inp"]
                )
    
    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / (torch.std(x, axis=1, keepdim=True)+1e-8)
    ############# diff from original paper ###################
    # becuase MoE has different intermiedate size for dense layer, expert and share expert
    # the feature channels are standarlized within each group
    # the original paper also does this way for attn and mlp
    ##########################################################
    fluc_metric = []
    expert_fluc_metric = standarlization(torch.stack(expert_fluc_metric_list))
    fluc_metric.append(expert_fluc_metric.view(-1))
    if isinstance(mlp, (DeepseekV2MoE, DeepseekV3MoE)):
        dense_fluc_metric = standarlization(torch.stack(dense_fluc_metric_list))
        fluc_metric.append(dense_fluc_metric.view(-1))
        shared_fluc_metric = standarlization(torch.stack(shared_fluc_metric_list))
        fluc_metric.append(shared_fluc_metric.view(-1))

    fluc_metric = torch.cat(fluc_metric)
    sorted_prune, _ = torch.sort(fluc_metric, descending=True)
    threshold = sorted_prune[math.ceil(len(sorted_prune)*args.preserve_channels_in_percent)]
    expert_mask = expert_fluc_metric > threshold
    if isinstance(mlp, (DeepseekV2MoE, DeepseekV3MoE)):
        dense_mask = dense_fluc_metric > threshold
        shared_mask = shared_fluc_metric > threshold

    dense_offset, expert_offset, shared_offset = 0, 0, 0
    flap_intermediate_sizes = {}
    for i in tqdm(range(num_layer), desc="pruning model weights"):
        mlp = model.model.layers[i].mlp
        if isinstance(mlp, (DeepseekV2MLP, DeepseekV3MLP)):
            weight_mask = dense_mask[dense_offset]
            valid_channels = torch.sum(weight_mask).item()
            if valid_channels == 0:
                logging.warn(f"layer {i}'s DENSE layer should have been fully pruned. We leave one channel for the sake of compatibility")
                weight_mask[torch.argmax(dense_fluc_metric[dense_offset])] = True
                valid_channels = 1

            # prune weight
            mlp.up_proj.weight.data = mlp.up_proj.weight.data[weight_mask]
            mlp.up_proj.out_features = valid_channels
            mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[weight_mask]
            mlp.gate_proj.out_features = valid_channels
            
            # attach bias
            output_bias = (dense_baseline_list[dense_offset] * ~weight_mask) @ mlp.down_proj.weight.t().float()
            mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, weight_mask]
            mlp.down_proj.bias = torch.nn.Parameter(output_bias.to(mlp.down_proj.weight.dtype))
            mlp.down_proj.in_features = valid_channels

            # update config
            flap_intermediate_sizes[i] = valid_channels
            dense_offset += 1

        elif isinstance(mlp, (DeepseekV2MoE, DeepseekV3MoE, OlmoeSparseMoeBlock)):
            num_experts = len(mlp.experts)
            flap_intermediate_sizes[i] = {
                "routed": [],
                "shared": 0,
            }
            for e_i in range(num_experts):
                weight_mask = expert_mask[expert_offset]
                valid_channels = torch.sum(weight_mask).item()
                if valid_channels == 0:
                    logging.warn(f"layer {i}'s expert {e_i} of the MoE layer should have been fully pruned. We leave one channel for the sake of compatibility")
                    weight_mask[torch.argmax(expert_fluc_metric[expert_offset])] = True
                    valid_channels = 1
                
                # prune weight
                mlp.experts[e_i].up_proj.weight.data = mlp.experts[e_i].up_proj.weight.data[weight_mask]
                mlp.experts[e_i].up_proj.out_features = valid_channels
                mlp.experts[e_i].gate_proj.weight.data = mlp.experts[e_i].gate_proj.weight.data[weight_mask]
                mlp.experts[e_i].gate_proj.out_features = valid_channels
                
                # attach bias
                output_bias = (expert_baseline_list[expert_offset] * ~weight_mask) @ mlp.experts[e_i].down_proj.weight.t().float()
                nan_count = torch.isnan(output_bias).sum()
                if nan_count > 0:
                    logging.warn(f"layer {i} expert {e_i} bias has {nan_count} NaN elements, valid channels: {valid_channels}")
                mlp.experts[e_i].down_proj.weight.data = mlp.experts[e_i].down_proj.weight.data[:, weight_mask]
                mlp.experts[e_i].down_proj.bias = torch.nn.Parameter(output_bias.to(dtype=mlp.experts[e_i].down_proj.weight.dtype))
                mlp.experts[e_i].down_proj.in_features = valid_channels

                flap_intermediate_sizes[i]['routed'].append(valid_channels)
                expert_offset += 1

            if isinstance(mlp, (DeepseekV2MoE, DeepseekV3MoE)):
                weight_mask = shared_mask[shared_offset]
                valid_channels = torch.sum(weight_mask).item()
                if valid_channels == 0:
                    logging.warn(f"layer {i}'s shared experts of the MoE layer should have been fully pruned. We leave one channel for the sake of compatibility")
                    weight_mask[torch.argmax(shared_fluc_metric[shared_offset])] = True
                    valid_channels = 1

                # prune weight
                mlp.shared_experts.up_proj.weight.data = mlp.shared_experts.up_proj.weight.data[weight_mask]
                mlp.shared_experts.up_proj.out_features = valid_channels
                mlp.shared_experts.gate_proj.weight.data = mlp.shared_experts.gate_proj.weight.data[weight_mask]
                mlp.shared_experts.gate_proj.out_features = valid_channels
                
                # attach bias
                output_bias = (shared_baseline_list[shared_offset] * ~weight_mask) @ mlp.shared_experts.down_proj.weight.t().float()
                mlp.shared_experts.down_proj.weight.data = mlp.shared_experts.down_proj.weight.data[:, weight_mask]
                mlp.shared_experts.down_proj.bias = torch.nn.Parameter(output_bias.to(dtype=mlp.shared_experts.down_proj.weight.dtype))
                mlp.shared_experts.down_proj.in_features = valid_channels

                # update config
                flap_intermediate_sizes[i]["shared"] = valid_channels
                shared_offset += 1
    
    # clear handles before saving
    for handle in handles:
        handle.remove()    
    
    # sanity check
    assert dense_offset == len(dense_baseline_list)
    assert expert_offset == len(expert_baseline_list)
    assert shared_offset == len(shared_baseline_list)

    # collect config
    model.config.flap_intermediate_sizes = flap_intermediate_sizes

    model.cpu()
    return model


@torch.no_grad()
def weight_prune_by_sparse_gpt(args, model, train_dataloader):
    model.cuda()
    device = torch.cuda.current_device()

    handles = []
    num_layer = model.config.num_hidden_layers

    cache = {}
    def create_hook(linear_name):
        def stateful_hook(module, _input, _output):
            inp = _input[0]
            batch_size = inp.shape[0]
            inp = inp.view(-1, inp.shape[-1]).t()

            num_samples = cache[linear_name]["num_samples"]
            H = cache[linear_name]["H"].to(device)
            del cache[linear_name]["H"]

            H *= num_samples / (num_samples + batch_size)
            inp = math.sqrt(2 / (num_samples + batch_size)) * inp.float()
            H += inp.matmul(inp.t())

            # write back stats
            cache[linear_name]["num_samples"] = num_samples + batch_size
            cache[linear_name]["H"] = H.to('cpu')
            del H

        return stateful_hook
    
    # register hook
    for i in range(num_layer):
        mlp = model.model.layers[i].mlp
        if isinstance(mlp, (DeepseekV2MLP, DeepseekV3MLP)):
            for name, layer in mlp.named_children():
                if name in ['gate_proj', 'up_proj', 'down_proj']:
                    inp_dim_size = layer.weight.shape[1]
                    module_name = f"layers.{i}.mlp.{name}"
                    cache[module_name] = {
                        "H": torch.zeros((inp_dim_size, inp_dim_size), device='cpu'),
                        "num_samples": 0
                    }
                    handle = layer.register_forward_hook(create_hook(module_name))
                    handles.append(handle)
        elif isinstance(mlp, (DeepseekV2MoE, DeepseekV3MoE, OlmoeSparseMoeBlock)):
            num_experts = len(mlp.experts)
            for e_i in range(num_experts):
                for name, layer in mlp.experts[e_i].named_children():
                    if name in ['gate_proj', 'up_proj', 'down_proj']:
                        inp_dim_size = layer.weight.shape[1]
                        module_name = f"layers.{i}.mlp.experts.{e_i}.{name}"
                        cache[module_name] = {
                            "H": torch.zeros((inp_dim_size, inp_dim_size), device='cpu'),
                            "num_samples": 0
                        }
                        handle = layer.register_forward_hook(create_hook(module_name))
                        handles.append(handle)

            if isinstance(mlp, (DeepseekV2MoE, DeepseekV3MoE)):
                for name, layer in mlp.shared_experts.named_children():
                    if name in ['gate_proj', 'up_proj', 'down_proj']:
                        shared_inp_dim_size = layer.weight.shape[1]
                        shared_module_name = f"layers.{i}.mlp.shared_experts.{name}"
                        cache[shared_module_name] = {
                            "H": torch.zeros((shared_inp_dim_size, shared_inp_dim_size), device='cpu'),
                            "num_samples": 0
                        }
                        handle = layer.register_forward_hook(create_hook(shared_module_name))
                        handles.append(handle)
        else:
            raise ValueError(f"unknow mlp type: {type(mlp)} at layer {i}")

    
    data_iter = iter(train_dataloader)
    for step in tqdm(range(args.max_steps), desc="collecting accumulated stats"):
        batch = next(data_iter)
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    for h in handles:
        h.remove()

    def _prune_weight(module, module_name_prefix):
        for name, layer in module.named_children():
            if name in ['gate_proj', 'up_proj', 'down_proj']:
                module_name = f"{module_name_prefix}.{name}"
                W = layer.weight.data.clone().float()
                H = cache[module_name]['H'].to(device)
                del cache[module_name]['H']
                dead = torch.diag(H) == 0
                H[dead, dead] = 1
                W[:, dead] = 0

                Losses = torch.zeros(W.shape[0], device=device)

                damp = args.percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(W.shape[1], device=device)
                H[diag, diag] += damp
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H

                for i1 in range(0, W.shape[1], args.sparsegpt_block_size):
                    i2 = min(i1 + args.sparsegpt_block_size, W.shape[1])
                    count = i2 - i1

                    W1 = W[:, i1:i2].clone()
                    Q1 = torch.zeros_like(W1)
                    Err1 = torch.zeros_like(W1)
                    Losses1 = torch.zeros_like(W1)
                    Hinv1 = Hinv[i1:i2, i1:i2]
                    mask1 = torch.zeros_like(W1) == 1

                    for i in range(count):
                        w = W1[:, i]
                        d = Hinv1[i, i]

                        if args.sparsegpt_prunen != 0 and i % args.sparsegpt_prunem == 0:
                            tmp = W1[:, i:(i + args.sparsegpt_prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + args.sparsegpt_prunem)].reshape((1, -1))) ** 2
                            mask1.scatter_(1, i + torch.topk(tmp, args.sparsegpt_prunen, dim=1, largest=False)[1], True)

                        q = w.clone()
                        q[mask1[:, i]] = 0

                        Q1[:, i] = q
                        Losses1[:, i] = (w - q) ** 2 / d ** 2

                        err1 = (w - q) / d
                        W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                        Err1[:, i] = err1

                    W[:, i1:i2] = Q1
                    Losses += torch.sum(Losses1, 1) / 2

                    W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

                # update weight# prune weight
                layer.weight.data = W.reshape(layer.weight.shape).to(layer.weight.data.dtype)
                del H

    for i in tqdm(range(num_layer), desc=f"pruning layer {i}"):
        mlp = model.model.layers[i].mlp
        if isinstance(mlp, (DeepseekV2MLP, DeepseekV3MLP)):
            _prune_weight(module=mlp, module_name_prefix=f"layers.{i}.mlp")
        elif isinstance(mlp, (DeepseekV2MoE, DeepseekV3MoE, OlmoeSparseMoeBlock)):
            num_experts = len(mlp.experts)
            for e_i in range(num_experts):
                _prune_weight(module=mlp.experts[e_i], module_name_prefix=f"layers.{i}.mlp.experts.{e_i}")
            if isinstance(mlp, (DeepseekV2MoE, DeepseekV3MoE)):
                _prune_weight(module=mlp.shared_experts, module_name_prefix=f"layers.{i}.mlp.shared_experts")
    model.cpu()
    return model

def weight_prune(args, model, train_dataloader):
    if args.weight_prune_metric == 'norm':
        return weight_prune_by_norm(args, model)
    elif args.weight_prune_metric == 'flap':
        return weight_prune_by_flap(args, model, train_dataloader)
    elif args.weight_prune_metric == 'sparsegpt':
        return weight_prune_by_sparse_gpt(args, model, train_dataloader)