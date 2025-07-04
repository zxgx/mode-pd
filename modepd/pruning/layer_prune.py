from copy import deepcopy
import math

import torch

from transformers import AutoModelForCausalLM

@torch.no_grad()
def layer_prune_cosine(args, model, train_dataloader):
    model.cuda()
    device = torch.cuda.current_device()
    num_layers = model.config.num_hidden_layers
    similarities = torch.zeros(num_layers, num_layers, device=device)
    denominator = torch.zeros(num_layers, num_layers, device=device)
    cache, handles = [None for _ in range(num_layers+1)], []

    # register hooks which save the input for each layer and the output of the last layer
    for i in range(num_layers):
        layer = model.model.layers[i]
        
        # bind layer index i into this hook to record input/outputs
        # Use a helper function to create a new scope for each hook
        def create_hook(layer_idx):
            def stateful_hook(module, _input, _output):
                cache[layer_idx] = _input[0].squeeze(0)
                if layer_idx == num_layers-1:
                    cache[-1] = _output[0].squeeze(0)

                    # accumulate results 
                    for j in range(num_layers):
                        for k in range(j+1, num_layers+1):
                            sim = torch.nn.functional.cosine_similarity(cache[j], cache[k], dim=-1).float()
                            denominator[j, k-j-1] += sim.shape[0]
                            similarities[j, k-j-1] += sim.sum()
                        cache[j] = None
            return stateful_hook
        
        # Create a unique hook for each layer with its correct index
        handle = layer.register_forward_hook(create_hook(i))
        handles.append(handle)
    
    # execute model to collect similarities
    for step, batch in enumerate(train_dataloader):
        if step == args.max_steps:
            break
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    similarities /= denominator

    # clear handles before saving
    for handle in handles:
        handle.remove()
    del cache

    # prune the model
    similarities_drop_1 = similarities[:, 0].view(-1)
    sorted_similarities, sorted_layer_id = torch.sort(similarities_drop_1, dim=0, descending=True)
    dropped_layer_list = sorted_layer_id[:args.drop_n_layers].tolist()

    reserved_layer_list = sorted(list(set(range(num_layers)) - set(dropped_layer_list)))
    layer_id_mapping = {}
    for new_id, reserved_old_id in enumerate(reserved_layer_list):
        layer_id_mapping[reserved_old_id] = new_id

    # save the pruned model state, this should not introduce more GPU memory usage
    model.cpu()
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

    # update config
    new_config = deepcopy(model.config)
    new_config.num_hidden_layers = len(layer_id_mapping)

    preserved_layers = sorted([int(s) for s in layer_id_mapping.keys()])

    # Model
    new_model = AutoModelForCausalLM.from_config(config=new_config)
    new_model.load_state_dict(save_state_dict, strict=True)  # update the layer parameters
    if not hasattr(new_model, "quantization_config"):
        new_model.bfloat16()

    return new_model


def layer_prune_angular(args, model, train_dataloader):
    model.cuda()
    device = torch.cuda.current_device()
    num_layers = model.config.num_hidden_layers
    
    similarities = torch.zeros(num_layers - args.drop_n_layers, device=device)
    cache, handles = [None for _ in range(num_layers)], []

    def create_hook(layer_idx):
        def stateful_hook(module, _input, _output):
            cache[layer_idx] = _input[0].squeeze(0).float()
            if layer_idx == num_layers - 1:
                for i in range(num_layers - args.drop_n_layers):
                    sim = torch.nn.functional.cosine_similarity(cache[i], cache[i+args.drop_n_layers], dim=-1)
                    similarities[i] += torch.sum(torch.acos(sim) / math.pi)
                    
        return stateful_hook

    # register hooks
    for i in range(num_layers):
        layer = model.model.layers[i]
        handle = layer.register_forward_hook(create_hook(i))
        handles.append(handle)
    
    # execute model
    for step, batch in enumerate(train_dataloader):
        if step == args.max_steps:
            break
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    # clear handles
    for handle in handles:
        handle.remove()


    # prune the model
    _, sorted_indices = torch.sort(torch.tensor(similarities), descending=False)
    dropped_layer_list = list(range(sorted_indices[0], sorted_indices[0]+args.drop_n_layers))

    reserved_layer_list = sorted(list(set(range(num_layers)) - set(dropped_layer_list)))
    layer_id_mapping = {}
    for new_id, reserved_old_id in enumerate(reserved_layer_list):
        layer_id_mapping[reserved_old_id] = new_id

    # save the pruned model state, this should not introduce more GPU memory usage
    model.cpu()
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

    # update config
    new_config = deepcopy(model.config)
    new_config.num_hidden_layers = num_layers-args.drop_n_layers

    # Model
    new_model = AutoModelForCausalLM.from_config(config=new_config)
    new_model.load_state_dict(save_state_dict, strict=True)  # update the layer parameters
    if not hasattr(new_model, "quantization_config"):
        new_model.bfloat16()

    return new_model


def layer_prune(args, model, train_dataloader):
    if args.layer_prune_metric == 'cosine':
        return layer_prune_cosine(args, model, train_dataloader)
    elif args.layer_prune_metric == 'angular':
        return layer_prune_angular(args, model, train_dataloader)