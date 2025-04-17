from tqdm import tqdm
from copy import deepcopy
import math
import logging

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from modepd.model.deepseek_v2.modeling_deepseek import DeepseekV2MLP
from modepd.model.moonshotai.modeling_deepseek import DeepseekV3MLP
from modepd.model.olmoe.modeling_olmoe import OlmoeMLP


@torch.no_grad()
def expert_prune_by_routing_score(args, model, train_dataloader):
    # Move the model to the GPU 
    model.cuda()
    # Retrieves the index of the currently active GPU device
    device = torch.cuda.current_device()

    handles = []
    scores, denominator = {}, {}

    # Get MoE model info
    num_layers = model.config.num_hidden_layers
    if "deepseek" in model.config.model_type:
        num_experts = model.config.n_routed_experts
    elif "olmoe" in model.config.model_type:
        num_experts = model.config.num_experts
    else:
        raise ValueError(f"unknow model type: {model.config.model_type}")
    
    # Identify MoE layer
    if "deepseek" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (
                model.config.n_routed_experts is not None and
                layer_idx >= model.config.first_k_dense_replace and 
                layer_idx % model.config.moe_layer_freq == 0
            )
        ]
    elif "olmoe" in model.config.model_type:
        valid_moe_layer_indices = list(range(num_layers))
    
    # Register forward hooks
    for i in valid_moe_layer_indices:
        layer = model.model.layers[i] # DeepseekV2DecoderLayer with MoE layer and number of experts > preserve_n_experts

        def create_hook(layer_idx):
            def stateful_hook(module, _input, _output):
                batch_size = _input[0].shape[0]

                if 'deepseek' in model.config.model_type:
                    topk_idx, topk_weight = _output[:2]
                elif 'olmoe' in model.config.model_type:
                    router_logits = _output
                    topk_weight = F.softmax(router_logits, dim=1, dtype=torch.float)
                    topk_weight, topk_idx = torch.topk(topk_weight, model.config.num_experts_per_tok, dim=-1)

                routing_weights = torch.zeros(
                    (topk_weight.shape[0], num_experts),
                    device=topk_weight.device,
                    dtype=torch.float
                )
                routing_weights = torch.scatter(routing_weights, dim=1, index=topk_idx, src=topk_weight)

                if layer_idx not in scores:
                    denominator[layer_idx] = batch_size
                    scores[layer_idx] = routing_weights.float().sum(0)
                else:
                    denominator[layer_idx] += batch_size
                    scores[layer_idx] += routing_weights.float().sum(0)

            return stateful_hook

        # Get MoE gate
        moe_gate_layer = layer.mlp.gate

        # register forward hook
        handle = moe_gate_layer.register_forward_hook(create_hook(i))
        handles.append(handle)

    # Execute model
    data_iter = iter(train_dataloader)
    for step in tqdm(range(args.max_steps), desc="collecting accumulated routing scores"):
        batch = next(data_iter)
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
    
    # clear handles before saving
    for handle in handles:
        handle.remove()

    # save the pruned model state, this should not introduce more GPU memory usage
    model.cpu()
    state_dict = model.state_dict()

    experts_to_keep_idx_dict = {}
    if args.expert_ranking_scope == 'layer':
        for layer_idx in scores.keys():
            # Calculate mean score
            score = scores[layer_idx] / denominator[layer_idx]
            
            # Get topK experts 
            _, experts_to_keep_idx = torch.topk(
                score,
                args.preserve_n_experts,
                largest=True
            )
            experts_to_keep_idx_dict[layer_idx] = sorted(experts_to_keep_idx.tolist())
    else:
        metric = torch.cat(list(scores.values()))
        sorted_scores, _ = torch.sort(metric, descending=True)
        threshold = sorted_scores[math.ceil(len(metric)*args.preserve_n_experts/num_experts)]
        for layer_idx in scores.keys():
            experts_to_keep_idx_dict[layer_idx] = sorted((torch.where(scores[layer_idx]>threshold)[0]).tolist())

    new_routed_experts = {}
    for layer_idx in valid_moe_layer_indices:
        experts_to_keep_idx = experts_to_keep_idx_dict[layer_idx]
        if len(experts_to_keep_idx) == 0:
            logging.warn(f"experts of layer {layer_idx} should have been fully removed. We preserve one for compatibility")
            experts_to_keep_idx.append(torch.argmax(scores[layer_idx]).item())
        new_routed_experts[layer_idx] = len(experts_to_keep_idx)

        # Update expert weight
        for expert_idx, old_expert_idx  in enumerate(experts_to_keep_idx):
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.experts.{old_expert_idx}.gate_proj.weight"]
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.experts.{old_expert_idx}.up_proj.weight"]
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.experts.{old_expert_idx}.down_proj.weight"]
        # Remove pruned experts
        for removed_expert_idx in range(len(experts_to_keep_idx), num_experts):
            del state_dict[f"model.layers.{layer_idx}.mlp.experts.{removed_expert_idx}.gate_proj.weight"]
            del state_dict[f"model.layers.{layer_idx}.mlp.experts.{removed_expert_idx}.up_proj.weight"]
            del state_dict[f"model.layers.{layer_idx}.mlp.experts.{removed_expert_idx}.down_proj.weight"]
        # Update MoE gate weight
        state_dict[f"model.layers.{layer_idx}.mlp.gate.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.gate.weight"][experts_to_keep_idx]
        if 'deepseek_v3' in model.config.model_type:
            state_dict[f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"] = state_dict[f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"][experts_to_keep_idx]

    new_config = deepcopy(model.config)
    if "deepseek" in model.config.model_type:
        new_config.n_routed_experts = new_routed_experts
    elif "olmoe" in model.config.model_type:
        new_config.num_experts = new_routed_experts
    new_model = AutoModelForCausalLM.from_config(config=new_config)
    # Model
    new_model.load_state_dict(state_dict, strict=True)  # update the layer parameters
    if not hasattr(new_model, "quantization_config"):
        new_model.bfloat16()

    return new_model


@torch.no_grad()
def align_expert_weight(reference_mlp, target_mlp):
    from scipy.optimize import linear_sum_assignment

    lsa_cost_matrix = torch.mm(
        reference_mlp.gate_proj.weight.data.float(), target_mlp.gate_proj.weight.data.float().t()
    )
    lsa_cost_matrix += torch.mm(
        reference_mlp.up_proj.weight.data.float(), target_mlp.up_proj.weight.data.float().t()
    )
    lsa_cost_matrix += torch.mm(
        reference_mlp.down_proj.weight.data.float().t(), target_mlp.down_proj.weight.data.float()
    )
    _, perm = linear_sum_assignment(lsa_cost_matrix.cpu().numpy(), maximize=True)


    d_ff = target_mlp.gate_proj.out_features

    # Check the permutation vector
    if perm.shape != (d_ff,):
        raise ValueError(f"The shape of the permutation vector should be (d_ff, ), but got {perm.shape}.")

    # Permute the weights of the MLP
    target_mlp.gate_proj.weight.data = target_mlp.gate_proj.weight.data[perm, :]
    target_mlp.up_proj.weight.data = target_mlp.up_proj.weight.data[perm, :]
    target_mlp.down_proj.weight.data = target_mlp.down_proj.weight.data[:, perm]

    return target_mlp


@torch.no_grad()
def expert_prune_by_mc_smoe(args, model, train_dataloader):
    # Move the model to the GPU 
    model.cuda()
    # Retrieves the index of the currently active GPU device
    device = torch.cuda.current_device()

    handles = []
    # Get MoE model info
    num_layers = model.config.num_hidden_layers
    if "deepseek" in model.config.model_type:
        num_experts = model.config.n_routed_experts
    elif "olmoe" in model.config.model_type:
        num_experts = model.config.num_experts
    else:
        raise ValueError(f"unknow model type: {model.config.model_type}")
    
    # Identify MoE layer
    if "deepseek" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (
                model.config.n_routed_experts is not None and
                layer_idx >= model.config.first_k_dense_replace and 
                layer_idx % model.config.moe_layer_freq == 0
            )
        ]
    elif "olmoe" in model.config.model_type:
        valid_moe_layer_indices = list(range(num_layers))
    
    # step 1: mlp weight permutation to align expert weight channels
    for i in tqdm(valid_moe_layer_indices, desc="Expert weight permuation"):
        layer = model.model.layers[i]
        for expert_idx in range(1, num_experts):
            layer.mlp.experts[expert_idx] = align_expert_weight(
                layer.mlp.experts[0], 
                layer.mlp.experts[expert_idx],
            )
    
    # step 2: merge experts according to access frequency and activation similarity
    sim_matrix = {}
    access_frequency = {}
    for i in valid_moe_layer_indices:
        layer = model.model.layers[i]
        sim_matrix[i] = torch.zeros(
            num_experts, num_experts, device=device, dtype=torch.float
        ) + torch.eye(num_experts, device=device, dtype=torch.float)
        access_frequency[i] = torch.zeros(
            num_experts, device=device, dtype=torch.float
        )
        def create_similarity_and_usage_hook(layer_idx):
            def stateful_hook(module, _input, _output):
                # _compute_all_similarities_by_router_logits
                hidden_states = _input[0]
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                # bs*seq_len, num_expert
                scores = F.sigmoid(
                    F.linear(hidden_states.float(), module.weight.float())
                )
                for e_i in range(num_experts-1):
                    for e_j in range(e_i+1, num_experts):
                        weight_i = scores[:, e_i].flatten()
                        weight_j = scores[:, e_j].flatten()
                        sim_matrix[layer_idx][e_i, e_j] = (F.cosine_similarity(
                            weight_i, weight_j, dim=-1, eps=1e-7
                        ) + 1) / 2

                # compute_all_usages
                if 'deepseek' in model.config.model_type:
                    topk_indices = _output[0].view(-1)
                elif 'olmoe' in model.config.model_type:
                    router_logits = _output
                    topk_weight = F.softmax(router_logits, dim=1, dtype=torch.float)
                    _, topk_idx = torch.topk(topk_weight, model.config.num_experts_per_tok, dim=-1)
                    topk_indices = topk_idx.view(-1)
                    
                access_frequency[layer_idx].scatter_add_(0, topk_indices, torch.ones_like(topk_indices, dtype=torch.float))
                access_frequency[layer_idx] = access_frequency[layer_idx] / torch.sum(access_frequency[layer_idx])

            return stateful_hook
        
        handle = layer.mlp.gate.register_forward_hook(create_similarity_and_usage_hook(i))
        handles.append(handle)

    # update sim_matrix and access_frequency
    data_iter = iter(train_dataloader)
    for step in tqdm(range(args.max_steps), desc="collecting similarities"):
        batch = next(data_iter)
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    # group_experts_into_clusters_by_routing_guided_globally
    # _assign_num_groups_per_layer
    total_num_groups = args.preserve_n_experts * len(valid_moe_layer_indices)
    all_usage_frequency = []
    usage_frequency_dict = deepcopy(access_frequency)
    for i in valid_moe_layer_indices:
        print(f">>> layer {i}:\naccess_frequency:\n{access_frequency[i]}\nsim_matrix:\n{sim_matrix[i]}")
        max_usage_index = torch.argmax(usage_frequency_dict[i])
        usage_frequency_dict[i][max_usage_index] = 1.0
        all_usage_frequency.append(usage_frequency_dict[i])

    all_usage_frequency = torch.cat(all_usage_frequency, dim=0)
    sorted_usage_frequency, sorted_indices = torch.sort(all_usage_frequency, descending=True)
    frequency_threshold = sorted_usage_frequency[total_num_groups]

    num_groups_per_layer = dict()
    for i in valid_moe_layer_indices:
        num_groups_per_layer[i] = torch.sum(
            (usage_frequency_dict[i]>frequency_threshold).long()
        ).item()
        print(f">>> layer {i} group: {num_groups_per_layer[i]}")
    
    core_experts = dict()
    group_state_dict = dict()
    for i in tqdm(valid_moe_layer_indices, desc="grouping experts layer by layer"):
        group_state_dict[i] = torch.arange(num_experts, device=device)
        # Assign top-K most-used experts with label 0 to K-1 respectively
        num_groups = num_groups_per_layer[i]
        group_member_count = torch.zeros(num_groups)
        indices_sorted_by_usage = torch.argsort(access_frequency[i], descending=True)
        core_expert_indices = indices_sorted_by_usage[:num_groups]
        core_experts[i] = core_expert_indices.tolist()
        for g_idx in range(num_groups):
            group_member_count[g_idx] += 1
            group_state_dict[i][core_expert_indices[g_idx]] = g_idx
        
        similarity_matrix = sim_matrix[i]
        for r_idx in range(num_groups, num_experts):
            expert_idx = indices_sorted_by_usage[r_idx]
            most_similar_core = core_expert_indices[
                torch.argmax(similarity_matrix[expert_idx, core_expert_indices])
            ]
            most_similar_group_label = group_state_dict[i][most_similar_core]
            group_state_dict[i][expert_idx] = most_similar_group_label
            group_member_count[most_similar_group_label] += 1
            if group_member_count[group_state_dict[i][expert_idx]] >= num_experts:
                raise ValueError(
                    f"group_member_count[group_state_dict[i][expert_idx]]={group_member_count[group_state_dict[i][expert_idx]]} >= num_experts={num_experts}")
        
        print(f"layer {i} group_member_count: {group_member_count}")
    
    # merge_by_groups_with_usage_frequency_weighting
    # TODO: step 3: compress experts
    new_num_experts = {}
    for i in tqdm(valid_moe_layer_indices, desc="merging experts layer by layer"):
        group_labels = group_state_dict[i]
        usage_frequencies = usage_frequency_dict[i]
        mlp = model.model.layers[i].mlp
        new_weights = []
        for label in group_labels.unique():
            expert_indices = torch.where(group_labels == label)[0]
            gate_proj_weight_list = torch.stack([
                mlp.experts[expert_idx].gate_proj.weight * usage_frequencies[expert_idx] \
                    for expert_idx in expert_indices
            ], dim=0)
            gate_proj_weight = torch.sum(gate_proj_weight_list, dim=0)/(
                torch.sum(usage_frequencies[expert_indices], dim=0) + 1e-7
            )

            up_proj_weight_list = torch.stack([
                mlp.experts[expert_idx].up_proj.weight * usage_frequencies[expert_idx] \
                    for expert_idx in expert_indices
            ], dim=0)
            up_proj_weight = torch.sum(up_proj_weight_list, dim=0) / (
                torch.sum(usage_frequencies[expert_indices], dim=0) + 1e-7
            )

            down_proj_weight_list = torch.stack([
                mlp.experts[expert_idx].down_proj.weight * usage_frequencies[expert_idx] \
                    for expert_idx in expert_indices
            ], dim=0)
            down_proj_weight = torch.sum(down_proj_weight_list, dim=0) / (
                torch.sum(usage_frequencies[expert_indices], dim=0) + 1e-7
            )

            gate_weight = torch.sum(
                (mlp.gate.weight[expert_indices] * usage_frequencies[expert_indices].unsqueeze(1) / torch.sum(usage_frequencies[expert_indices]))
            , dim=0, keepdim=True)
            new_weights.append({
                "gate_proj": gate_proj_weight,
                "up_proj": up_proj_weight,
                "down_proj": down_proj_weight,
                "gate_weight": gate_weight,
            })
            if hasattr(mlp.gate, "e_score_correction_bias"):
                expert_weight = torch.sum(
                    mlp.gate.e_score_correction_bias[expert_indices] * usage_frequencies[expert_indices] / torch.sum(usage_frequencies[expert_indices])
                , dim=0, keepdim=True)
                new_weights[-1]["expert_weight"] = expert_weight
        
        num_expert_this_layer = len(new_weights)
        new_num_experts[i] = num_expert_this_layer

        mlp.experts = mlp.experts[:num_expert_this_layer]
        for e_idx, weights in enumerate(new_weights):
            mlp.experts[e_idx].gate_proj.weight.copy_(weights["gate_proj"])
            mlp.experts[e_idx].up_proj.weight.copy_(weights["up_proj"])
            mlp.experts[e_idx].down_proj.weight.copy_(weights["down_proj"])

        mlp.gate.weight.data = torch.cat([each['gate_weight'] for each in new_weights], dim=0)
        if hasattr(mlp.gate, "e_score_correction_bias"):
            mlp.gate.e_score_correction_bias.data = torch.cat([each['expert_weight'] for each in new_weights])
    
    # clear handles before saving
    for handle in handles:
        handle.remove()

    model.cpu()
    state_dict = model.state_dict()

    new_config = deepcopy(model.config)
    if "deepseek" in model.config.model_type:
        new_config.n_routed_experts = new_num_experts
    elif "olmoe" in model.config.model_type:
        new_config.num_experts = new_num_experts
    new_model = AutoModelForCausalLM.from_config(config=new_config)

    # Model
    new_model.load_state_dict(state_dict, strict=True)  # update the layer parameters
    if not hasattr(new_model, "quantization_config"):
        new_model.bfloat16()

    return new_model


@torch.no_grad()
def expert_prune_by_mone(args, model, train_dataloader):
    # Move the model to the GPU 
    model.cuda()
    # Retrieves the index of the currently active GPU device
    device = torch.cuda.current_device()

    handles = []
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.moe_intermediate_size
    if "deepseek_v3" in model.config.model_type:
        novice_cls = DeepseekV3MLP
        num_experts = model.config.n_routed_experts
    elif "deepseek_v2" in model.config.model_type:
        novice_cls = DeepseekV2MLP
        num_experts = model.config.n_routed_experts
    elif "olmoe" in model.config.model_type:
        novice_cls = OlmoeMLP
        num_experts = model.config.num_experts
    else:
        raise ValueError(f"unknow model type: {model.config.model_type}")

    # Identify MoE layer
    if "deepseek" in model.config.model_type:
        valid_moe_layer_indices = [
            layer_idx for layer_idx in range(num_layers) 
            if (
                model.config.n_routed_experts is not None and
                layer_idx >= model.config.first_k_dense_replace and 
                layer_idx % model.config.moe_layer_freq == 0
            )
        ]
    elif "olmoe" in model.config.model_type:
        valid_moe_layer_indices = list(range(num_layers))
    
    #########################################
    # Create hooks to collect pruning metrics
    bias_stats = {}
    def create_expert_hook(expert_name):
        def stateful_expert_hook(module, _input, _output):
            out = _output
            out = out.view(-1, out.shape[-1])
            inp = _input[0]
            inp = inp.view(-1, inp.shape[-1])
            token_size = out.shape[0]

            # retrieve stats
            num_tokens = bias_stats[expert_name]["num_tokens"]
            baseline_out = bias_stats[expert_name]["baseline_out"]
            if args.mone_ranking_metric in ['output_fluctuation', 'io_fluctuation', 'fusion']:
                fluc_out = bias_stats[expert_name]["fluc_out"]
            elif args.mone_ranking_metric in ['token_fluctuation', 'intermediate_fluctuation', 'rs_intermediate']:
                baseline_inp = bias_stats[expert_name]["baseline_inp"]
                fluc_inp = bias_stats[expert_name]["fluc_inp"]

            # update moving average and fluctuation
            baseline_out *= num_tokens / (num_tokens + token_size)
            baseline_out += torch.sum(out.float(), dim=0) / (num_tokens + token_size)
            if args.mone_ranking_metric in ['output_fluctuation', 'fusion']:
                if num_tokens > 0:
                    fluc_out *= (num_tokens - 1) / (num_tokens + token_size - 1)
                    fluc_out += torch.sum((out - baseline_out.unsqueeze(0)).float().pow(2), dim=0) / (num_tokens + token_size)
            elif args.mone_ranking_metric in ['io_fluctuation',]:
                if num_tokens > 0:
                    fluc_out *= (num_tokens - 1) / (num_tokens + token_size - 1)
                    inp = _input[0]
                    inp = inp.view(-1, inp.shape[-1])
                    fluc_out += torch.sum((inp - out).float().pow(2), dim=0) / (num_tokens + token_size)
            elif args.mone_ranking_metric in ['token_fluctuation', 'intermediate_fluctuation', 'rs_intermediate']:
                baseline_inp *= num_tokens / (num_tokens + token_size)
                baseline_inp += torch.sum(inp, dim=0) / (num_tokens + token_size)
                if num_tokens > 0:
                    fluc_inp *= (num_tokens - 1) / (num_tokens + token_size - 1)
                    fluc_inp += torch.sum((inp - baseline_inp.unsqueeze(0))**2, dim=0) / (num_tokens + token_size)
            
            # write back stats
            bias_stats[expert_name]["num_tokens"] += token_size
            bias_stats[expert_name]["baseline_out"] = baseline_out
            if args.mone_ranking_metric in ['output_fluctuation', 'io_fluctuation', 'fusion']:
                bias_stats[expert_name]['fluc_out'] = fluc_out
            elif args.mone_ranking_metric in ['token_fluctuation', 'intermediate_fluctuation', 'rs_intermediate']:
                bias_stats[expert_name]["baseline_inp"] = baseline_inp
                bias_stats[expert_name]["fluc_inp"] = fluc_inp
        
        return stateful_expert_hook

    if args.mone_ranking_metric in ['routing_score', 'fusion', 'rs_intermediate']:
        routing_stats = {}
        def create_gate_hook(layer_idx):
            def stateful_gate_hook(module, _input, _output):
                batch_size = _input[0].shape[0]
                if 'deepseek' in model.config.model_type:
                    topk_idx, topk_weight = _output[:2]
                elif 'olmoe' in model.config.model_type:
                    router_logits = _output
                    topk_weight = F.softmax(router_logits, dim=1, dtype=torch.float)
                    topk_weight, topk_idx = torch.topk(topk_weight, model.config.num_experts_per_tok, dim=-1)

                assert topk_idx.dim() == 2
                # token_size = topk_idx.shape[0]

                routing_weights = torch.zeros(
                    (topk_weight.shape[0], num_experts),
                    device=device, dtype=torch.float
                )
                # num_tokens_per_expert = torch.zeros_like(routing_weights)

                routing_weights = torch.scatter(routing_weights, dim=1, index=topk_idx, src=topk_weight)
                # num_tokens_per_expert = torch.scatter_add(num_tokens_per_expert, dim=1, index=topk_idx, src=torch.ones_like(topk_weight))
                # num_tokens_per_expert = torch.sum(num_tokens_per_expert, dim=0)

                scores = routing_stats[layer_idx]["scores"]
                num_tokens = routing_stats[layer_idx]["num_tokens"]
                
                # scores *= num_tokens / (num_tokens + num_tokens_per_expert)
                # scores += torch.sum(routing_weights, dim=0) / (num_tokens + num_tokens_per_expert)
                
                scores *= num_tokens / (num_tokens + batch_size)
                scores += torch.sum(routing_weights, dim=0) / (num_tokens + batch_size)

                routing_stats[layer_idx]["num_tokens"] += batch_size #num_tokens_per_expert
                routing_stats[layer_idx]["scores"] = scores

            return stateful_gate_hook
        
    for i in valid_moe_layer_indices:
        mlp = model.model.layers[i].mlp
        if args.mone_ranking_metric in ['routing_score', 'fusion','rs_intermediate']:
            routing_stats[i] = {
                "scores": torch.zeros(num_experts, device=device, dtype=torch.float),
                "num_tokens": torch.zeros(num_experts, device=device, dtype=torch.float),
            }
            
            handle = mlp.gate.register_forward_hook(create_gate_hook(i))
            handles.append(handle)
            
        for e_idx in range(len(mlp.experts)):
            expert_name = f"layers.{i}.experts.{e_idx}"
            bias_stats[expert_name] = {
                "num_tokens": 0,
                "baseline_out": torch.zeros(hidden_size, device=device, dtype=torch.float),
            }

            if args.mone_ranking_metric in ['output_fluctuation', 'io_fluctuation', 'fusion']:
                bias_stats[expert_name]['fluc_out'] = torch.zeros(hidden_size, device=device, dtype=torch.float)
                handle = mlp.experts[e_idx].register_forward_hook(create_expert_hook(expert_name))
                handles.append(handle)
            elif args.mone_ranking_metric in ['token_fluctuation']:
                bias_stats[expert_name]["baseline_inp"] = torch.zeros(hidden_size, device=device, dtype=torch.float)
                bias_stats[expert_name]["fluc_inp"] = torch.zeros(hidden_size, device=device, dtype=torch.float)
                handle = mlp.experts[e_idx].register_forward_hook(create_expert_hook(expert_name))
                handles.append(handle)
            elif args.mone_ranking_metric in ['intermediate_fluctuation']:
                bias_stats[expert_name] = {
                    "num_tokens": 0,
                    "baseline_out": torch.zeros(hidden_size, device=device, dtype=torch.float),
                    "baseline_inp": torch.zeros(intermediate_size, device=device, dtype=torch.float),
                    "fluc_inp": torch.zeros(intermediate_size, device=device, dtype=torch.float),
                }
                handle = mlp.experts[e_idx].down_proj.register_forward_hook(create_expert_hook(expert_name))
                handles.append(handle)

    data_iter = iter(train_dataloader)
    for step in tqdm(range(args.max_steps), desc="collecting accumulated stats"):
        batch = next(data_iter)
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
    
    for handle in handles:
        handle.remove()
    
    #########################
    # Collect pruning metrics
    metric_list = {}
    if args.mone_ranking_metric == 'routing_score':
        for layer_idx in valid_moe_layer_indices:
            metric_list[layer_idx] = routing_stats[layer_idx]["scores"]
    elif args.mone_ranking_metric in ['output_fluctuation', 'io_fluctuation']:
        for layer_idx in valid_moe_layer_indices:
            fluc_list = [bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['fluc_out'] for e_idx in range(num_experts)]
            output_fluc = torch.stack(fluc_list)
            metric_list[layer_idx] = torch.norm(output_fluc, dim=1)
    elif args.mone_ranking_metric == 'fusion':
        for layer_idx in valid_moe_layer_indices:
            fluc_list = [bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['fluc_out'] for e_idx in range(num_experts)]
            # num_experts
            output_fluc = torch.norm(torch.sqrt(torch.stack(fluc_list)), dim=1)
            metric_list[layer_idx] = (args.fusion_io_weight * output_fluc) * ((1-args.fusion_io_weight) * routing_stats[layer_idx]["scores"])
    elif args.mone_ranking_metric=='token_fluctuation':
        for layer_idx in valid_moe_layer_indices:
            fluc_list = [bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['fluc_inp'] for e_idx in range(num_experts)]
            input_fluc = torch.stack(fluc_list)
            metric_list[layer_idx] = torch.norm(input_fluc, dim=1)
    elif args.mone_ranking_metric=='intermediate_fluctuation':
        for layer_idx in valid_moe_layer_indices:
            scores = []
            for e_idx in range(num_experts):
                inp_fluc = bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['fluc_inp']
                mlp = model.model.layers[layer_idx].mlp
                intermediate_score = inp_fluc * torch.sum(mlp.experts[e_idx].down_proj.weight.data.pow(2), dim=0)
                scores.append(intermediate_score)
            metric_list[layer_idx] = torch.norm(torch.stack(scores), dim=1)
    else:
        raise ValueError(f"unknow ranking metric: {args.mone_ranking_metric}")

    ###########################
    # Collect pruning threshold
    if args.expert_ranking_scope == 'model':
        metric = torch.cat(list(metric_list.values()))
        sorted_scores, _ = torch.sort(metric, descending=True)
        threshold_val = sorted_scores[math.ceil(len(sorted_scores)*args.preserve_n_experts/num_experts)]
        threshold = {layer_idx: threshold_val for layer_idx in valid_moe_layer_indices}
    else:
        threshold = {}
        for layer_idx in metric_list:
            metric = metric_list[layer_idx]
            sorted_scores, _ = torch.sort(metric, descending=True)
            threshold[layer_idx] = sorted_scores[args.preserve_n_experts]
    
    #################
    # Run pruning process
    approximate_experts = {}
    approximate_expert_init_tokens = {}
    for layer_idx in valid_moe_layer_indices:
        layer_metric = metric_list[layer_idx]
        expert_mask = layer_metric > threshold[layer_idx]

        expert_indicator_list = expert_mask.tolist()
        approximate_experts[layer_idx] = []
        approximate_expert_init_tokens[layer_idx] = []
        for expert_idx, is_preserved in enumerate(expert_indicator_list):
            if not is_preserved:
                approximate_experts[layer_idx].append(expert_idx)
                approximate_expert_init_tokens[layer_idx].append(
                    bias_stats[expert_name]['num_tokens'] if args.enable_novice_evolving else 0
                )
                expert_name = f"layers.{layer_idx}.experts.{expert_idx}"
                novice = novice_cls(model.config, is_approx=True, 
                    acc_tokens=bias_stats[expert_name]['num_tokens'] if args.enable_novice_evolving else 0)
                novice.approx_value.copy_(bias_stats[expert_name]["baseline_out"])                
                model.model.layers[layer_idx].mlp.experts[expert_idx] = novice.bfloat16()
                
    # update configs
    model.config.approximate_experts = approximate_experts
    model.config.approximate_expert_init_tokens = approximate_expert_init_tokens
    model.cpu()
    return model


def expert_prune(args, model, train_dataloader):    

    if args.expert_prune_metric == 'routing_score':
        return expert_prune_by_routing_score(args, model, train_dataloader)
    elif args.expert_prune_metric == 'mc_smoe':
        return expert_prune_by_mc_smoe(args, model, train_dataloader)
    elif args.expert_prune_metric == 'mone':
        return expert_prune_by_mone(args, model, train_dataloader)