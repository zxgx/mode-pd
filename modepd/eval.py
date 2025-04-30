import os
import logging
import argparse
from pprint import pformat
import json
import numpy as np

from modepd.utils import register_custom_model
register_custom_model()

import torch
import torch.nn.functional as F
import lm_eval
from lm_eval.models.huggingface import HFLM

from modepd.model.deepseek_v2.modeling_deepseek import DeepseekV2MLP
from modepd.model.moonshotai.modeling_deepseek import DeepseekV3MLP

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def _handle_non_serializable(o):
    """Copied from https://github.com/meta-llama/llama-recipes/blob/b5f64c0b69d7ff85ec186d964c6c557d55025969/tools/benchmarks/llm_eval_harness/eval.py#L18
    """
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def get_args():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--hf_model", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", type=torch.dtype, default=torch.bfloat16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_parallel", action="store_true")
    # lm_eval config
    parser.add_argument(
        "--tasks", type=str, nargs='+',
        default=[
            # English
            "mmlu", 
            # Chinese
            "cmmlu",
            # Math
            "gsm8k",
            # Code
            "humaneval",
            ])
    parser.add_argument(
        "--num_fewshots", type=int, nargs='+',
        default=[
            # English
            # "mmlu",
            5,
            # Chinese
            # "cmmlu",
            5,
            # Math
            # "gsm8k",
            8,
            # Code
            # "humaneval"
            0,
            ])
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--stats_output_dir", type=str, default=None)

    return parser.parse_args()


class Analyser:
    def __init__(self, model):
        self.device = torch.cuda.current_device()

        self.model = model
        self.hidden_size = model.config.hidden_size
        self.num_layers = model.config.num_hidden_layers

        if "deepseek" in model.config.model_type:
            self.valid_moe_layer_indices = [
                layer_idx for layer_idx in range(self.num_layers) 
                if (
                    model.config.n_routed_experts is not None and
                    layer_idx >= model.config.first_k_dense_replace and 
                    layer_idx % model.config.moe_layer_freq == 0
                )
            ]
            self.num_experts = model.config.n_routed_experts
        elif "olmoe" in model.config.model_type:
            self.valid_moe_layer_indices = list(range(self.num_layers))
            self.num_experts = model.config.num_experts

        self.handles = []
        self.build_hooks()

    def build_hooks(self):
        self.bias_stats = {}
        def create_expert_hook(expert_name):
            def stateful_expert_hook(module, _input, _output):
                out = _output
                out = out.view(-1, out.shape[-1])
                inp = _input[0]
                inp = inp.view(-1, inp.shape[-1])
                token_size = out.shape[0]

                if token_size == 0:
                    return

                # retrieve stats
                num_tokens = self.bias_stats[expert_name]["num_tokens"]
                baseline_out = self.bias_stats[expert_name]["baseline_out"]
                fluc_out = self.bias_stats[expert_name]["fluc_out"]
                
                # update moving average and fluctuation
                baseline_out *= num_tokens / (num_tokens + token_size)
                baseline_out += torch.sum(out.float(), dim=0) / (num_tokens + token_size)
                if num_tokens > 0:
                    fluc_out *= (num_tokens - 1) / (num_tokens + token_size - 1)
                    fluc_out += torch.sum((out - baseline_out.unsqueeze(0)).float().pow(2), dim=0) / (num_tokens + token_size)
                
                # write back stats
                self.bias_stats[expert_name]["num_tokens"] += token_size
                self.bias_stats[expert_name]["baseline_out"] = baseline_out
                self.bias_stats[expert_name]['fluc_out'] = fluc_out
            
            return stateful_expert_hook

        self.routing_stats = {}
        def create_gate_hook(layer_idx):
            def stateful_gate_hook(module, _input, _output):
                batch_size = _input[0].shape[0]
                if 'deepseek' in self.model.config.model_type:
                    topk_idx, topk_weight = _output[:2]
                elif 'olmoe' in self.model.config.model_type:
                    router_logits = _output
                    topk_weight = F.softmax(router_logits, dim=1, dtype=torch.float)
                    topk_weight, topk_idx = torch.topk(topk_weight, self.model.config.num_experts_per_tok, dim=-1)

                assert topk_idx.dim() == 2

                routing_weights = torch.zeros(
                    (topk_weight.shape[0], self.num_experts),
                    device=self.device, dtype=torch.float
                )

                routing_weights = torch.scatter(routing_weights, dim=1, index=topk_idx, src=topk_weight)

                scores = self.routing_stats[layer_idx]["scores"]
                num_tokens = self.routing_stats[layer_idx]["num_tokens"]

                scores *= num_tokens / (num_tokens + batch_size)
                scores += torch.sum(routing_weights, dim=0) / (num_tokens + batch_size)

                self.routing_stats[layer_idx]["num_tokens"] += batch_size #num_tokens_per_expert
                self.routing_stats[layer_idx]["scores"] = scores

            return stateful_gate_hook
        
        for i in self.valid_moe_layer_indices:
            mlp = self.model.model.layers[i].mlp
            self.routing_stats[i] = {
                "scores": torch.zeros(self.num_experts, device=self.device, dtype=torch.float),
                "num_tokens": torch.zeros(self.num_experts, device=self.device, dtype=torch.float),
            }
                
            handle = mlp.gate.register_forward_hook(create_gate_hook(i))
            self.handles.append(handle)
                
            for e_idx in range(self.num_experts):
                expert_name = f"layers.{i}.experts.{e_idx}"
                self.bias_stats[expert_name] = {
                    "num_tokens": 0,
                    "baseline_out": torch.zeros(self.hidden_size, device=self.device, dtype=torch.float),
                    "fluc_out": torch.zeros(self.hidden_size, device=self.device, dtype=torch.float)
                }

                handle = mlp.experts[e_idx].register_forward_hook(create_expert_hook(expert_name))
                self.handles.append(handle)
            
    def reset_stats(self):
        for i in self.valid_moe_layer_indices:
            mlp = self.model.model.layers[i].mlp

            self.routing_stats[i]["scores"].zero_()
            self.routing_stats[i]["num_tokens"].zero_()

            num_experts = len(mlp.experts)
            for e_idx in range(num_experts):
                expert_name = f"layers.{i}.experts.{e_idx}"
                self.bias_stats[expert_name]["num_tokens"] = 0
                self.bias_stats[expert_name]["baseline_out"].zero_()
                self.bias_stats[expert_name]["fluc_out"].zero_()

    def dump_stats(self, model_id, task, save_dir=None):
        freq_list, var_list = {}, {}
        for layer_idx in self.valid_moe_layer_indices:
            freq_list[layer_idx] = self.routing_stats[layer_idx]["scores"]
            data = [self.bias_stats[f'layers.{layer_idx}.experts.{e_idx}']['fluc_out'] for e_idx in range(self.num_experts)]
            var_list[layer_idx] = torch.norm(torch.sqrt(torch.stack(data)), dim=1)
        freq_list = torch.stack(list(freq_list.values()))
        var_list = torch.stack(list(var_list.values()))
        logging.info(f"{model_id} frequency map for task {task}: {freq_list}")
        logging.info(f"{model_id} variance map for task {task}: {var_list}")

        if save_dir is not None:
            torch.save(
                {"frequency": freq_list, "variance": var_list}, 
                os.path.join(save_dir, f"{model_id.replace('/', '_')}-{task}.pt")
            )


def main():
    args = get_args()
    logging.info(f"{pformat(vars(args), indent=2, width=120)}")

    hf_model = args.hf_model
    
    model_kwargs = {
        "trust_remote_code": True if args.trust_remote_code else None,
        # "torch_dtype": args.dtype,  # override by `dtype`` in lm_eval.HFLM
        "dtype": args.dtype,
        "device_map": "auto",
        "batch_size": args.batch_size, #"auto:4",
        "backend": "causal",
    }

    if "deepseek-ai/DeepSeek-V2.5-1210" == hf_model:
        # hardcoded according to the model card
        max_memory = {i: "75GB" for i in range(8)}
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "sequential",
            "dtype": torch.bfloat16,
            "max_memory": max_memory,
            "attn_implementation": "eager",
            "batch_size": "auto",
            "backend": "causal",
        }
    elif "gemma-3" in hf_model:
        model_kwargs['attn_implementation'] = 'eager'

    lm_eval_kwargs = {
        "limit": args.limit,
        "log_samples": False,
        "confirm_run_unsafe_code": True,
    }
    
    lm_obj = HFLM(hf_model, parallelize=args.model_parallel, **model_kwargs)
    lm_obj.model.cuda()
    lm_obj.model.config.use_cache = True
    lm_obj.model.generation_config.use_cache = True
    if "DeepSeek-V2" in hf_model:
        lm_obj.model.generation_config.pad_token_id = lm_obj.model.generation_config.eos_token_id
    logging.info(f"rank: {lm_obj.rank} / {lm_obj.world_size} model device: {lm_obj.model.device}, generation_config: {lm_obj.model.generation_config}, use_cache: {(lm_obj.model.generation_config.use_cache, lm_obj.model.config.use_cache)}")
    
    analyser = Analyser(lm_obj.model)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for task, num_fewshot in zip(args.tasks, args.num_fewshots):
        logging.info(f"Evaluate task: {task} with fewshot {num_fewshot}")
        results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=[task],
            num_fewshot=num_fewshot,
            **lm_eval_kwargs,
        )

        if args.output_dir:
            with open(os.path.join(args.output_dir, f"{hf_model.replace('/', '_')}-{task}.json"), "w") as f:
                json.dump(results, f, default=_handle_non_serializable, indent=2)
        logging.info(pformat(results))
        analyser.dump_stats(hf_model, task, args.stats_output_dir)
        analyser.reset_stats()


if __name__ == "__main__":
    main()
