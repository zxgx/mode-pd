from copy import deepcopy

import torch
from torch import nn
from transformers import AutoModelForCausalLM


@torch.no_grad()
def expert_prune(args, model, train_dataloader):
    
    # Move the model to the GPU 
    model.cuda()
    # Retrieves the index of the currently active GPU device
    device = torch.cuda.current_device()

    handles = []
    scores, denominator = {}, {}

    new_config = deepcopy(model.config)
    new_config.n_routed_experts = args.preserve_n_experts
    new_config.num_experts_per_tok = min(args.preserve_n_experts, new_config.num_experts_per_tok)
    new_model = AutoModelForCausalLM.from_config(config=new_config)

    # Get MoE model info
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.n_routed_experts
    # Identify MoE layer
    moe_layer_indices = [
        layer_idx for layer_idx in range(num_layers) 
        if (
            model.config.n_routed_experts is not None and
            layer_idx >= model.config.first_k_dense_replace and 
            layer_idx % model.config.moe_layer_freq == 0
        )
    ] 
    if isinstance(num_experts, list):
        valid_moe_layer_indices = [i for i in moe_layer_indices if num_experts[i] >= 0]
    else:
        valid_moe_layer_indices = moe_layer_indices
    
    # Register forward hooks
    for i in valid_moe_layer_indices:
        current_layer_num_experts = num_experts[i] if isinstance(num_experts, list) else num_experts
        if current_layer_num_experts > args.preserve_n_experts:
            layer = model.model.layers[i] # DeepseekV2DecoderLayer with MoE layer and number of experts > preserve_n_experts

            def create_hook(layer_idx):
                def stateful_hook(module, input, output):
                    # batch_size
                    if len(input[0].data.shape) == 2:
                        batch_size = 1
                    else:
                        batch_size = input[0].data.shape[0]

                    topk_idx, topk_weight = output[0].data, output[1].data
                    topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1])  # shape(batch_size * seq_len, num_selects)
                    topk_weight = topk_weight.reshape(-1, topk_weight.shape[-1])  # shape(batch_size * seq_len, num_selects)

                    routing_weights = torch.zeros(
                        (topk_weight.shape[0], module.n_routed_experts),
                        device=topk_weight.device,
                        dtype=topk_weight.dtype
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
    for step, batch in enumerate(train_dataloader):
        if step == args.max_steps:
            break
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    # save the pruned model state, this should not introduce more GPU memory usage
    model.cpu()
    state_dict = model.state_dict()

    for layer_idx in scores.keys():
        # Calculate mean score
        score = scores[layer_idx] / denominator[layer_idx]
        
        # Get topK experts 
        _, experts_to_keep_idx = torch.topk(
            score,
            args.preserve_n_experts,
            largest=True
        )
        experts_to_keep_idx = sorted(experts_to_keep_idx.tolist())

        # Update expert weight
        for old_expert_idx, expert_idx in zip(experts_to_keep_idx, range(args.preserve_n_experts)):
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.experts.{old_expert_idx}.gate_proj.weight"]
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.experts.{old_expert_idx}.up_proj.weight"]
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.experts.{old_expert_idx}.down_proj.weight"]
        # Remove pruned experts
        for reoved_expert_idx in range(args.preserve_n_experts, num_experts):
            del state_dict[f"model.layers.{layer_idx}.mlp.experts.{reoved_expert_idx}.gate_proj.weight"]
            del state_dict[f"model.layers.{layer_idx}.mlp.experts.{reoved_expert_idx}.up_proj.weight"]
            del state_dict[f"model.layers.{layer_idx}.mlp.experts.{reoved_expert_idx}.down_proj.weight"]
        # Update MoE gate weight
        state_dict[f"model.layers.{layer_idx}.mlp.gate.weight"] = state_dict[f"model.layers.{layer_idx}.mlp.gate.weight"][experts_to_keep_idx]
        state_dict[f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"] = state_dict[f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"][experts_to_keep_idx]
    
    # clear handles before saving
    for handle in handles:
        handle.remove()

    # Model
    new_model.load_state_dict(state_dict, strict=True)  # update the layer parameters
    if not hasattr(new_model, "quantization_config"):
        new_model.bfloat16()

    return new_model
