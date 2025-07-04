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
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM

from modepd.model.deepseek_v2.modeling_deepseek import DeepseekV2MLP
from modepd.model.moonshotai.modeling_deepseek import DeepseekV3MLP

# Add these imports for checkpoint loading
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    
    # Add checkpoint loading arguments
    parser.add_argument("--checkpoint_dir", type=str, default=None, 
                       help="Directory containing DeepSpeed checkpoint (e.g., step_100)")
    parser.add_argument("--checkpoint_tag", type=str, default=None,
                       help="DeepSpeed checkpoint tag (usually the step number)")
    
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
    parser.add_argument("--log_samples", action='store_true')
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--stats_output_dir", type=str, default=None)
    parser.add_argument("--analyse", action='store_true')
    # post-processing for think pattern
    parser.add_argument(
        "--postprocess_think_pattern", 
        action="store_true",
        help="Enable post-processing to extract answers from think pattern output"
    )
    parser.add_argument(
        "--think_start_pattern",
        type=str,
        default="<think>",
        help="Pattern that marks the start of thinking content"
    )
    parser.add_argument(
        "--think_end_pattern", 
        type=str,
        default="</think>",
        help="Pattern that marks the end of thinking content"
    )

    return parser.parse_args()


def load_model_from_checkpoint(model_name_or_path, checkpoint_dir, checkpoint_tag=None, dtype=torch.bfloat16):
    """
    Load model from DeepSpeed checkpoint directory.
    
    Args:
        model_name_or_path: Original model name or path (for tokenizer and config)
        checkpoint_dir: Path to DeepSpeed checkpoint directory
        checkpoint_tag: Optional checkpoint tag (step number)
        dtype: Model dtype
    
    Returns:
        model: Loaded model with checkpoint weights
        tokenizer: Tokenizer from original model
    """
    logging.info(f"Loading model from checkpoint: {checkpoint_dir}")
    
    # Load tokenizer and config from original model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # Initialize model with original config
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="cpu"  # Load on CPU first
    )
    
    # Load checkpoint weights
    checkpoint_files = []
    if checkpoint_tag:
        # Look for specific checkpoint tag
        mp_rank_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"mp_rank_{checkpoint_tag}")]
    else:
        # Look for any checkpoint files
        mp_rank_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("mp_rank")]
    
    if not mp_rank_files:
        # Try to find safetensors or pytorch_model files
        safetensor_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')]
        pytorch_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('pytorch_model') and f.endswith('.bin')]
        
        if safetensor_files:
            checkpoint_files = safetensor_files
        elif pytorch_files:
            checkpoint_files = pytorch_files
        else:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    else:
        checkpoint_files = mp_rank_files
    
    # Load state dict from checkpoint
    state_dict = {}
    for ckpt_file in checkpoint_files:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        logging.info(f"Loading checkpoint file: {ckpt_path}")
        
        if ckpt_file.endswith('.safetensors'):
            ckpt_state_dict = load_file(ckpt_path)
        else:
            ckpt_state_dict = torch.load(ckpt_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'module' in ckpt_state_dict:
            # DeepSpeed format
            for key, value in ckpt_state_dict['module'].items():
                state_dict[key] = value
        elif 'model_state_dict' in ckpt_state_dict:
            # Standard PyTorch format
            for key, value in ckpt_state_dict['model_state_dict'].items():
                state_dict[key] = value
        else:
            # Direct state dict
            for key, value in ckpt_state_dict.items():
                state_dict[key] = value
    
    # Load state dict into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logging.warning(f"Missing keys when loading checkpoint: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
    
    logging.info(f"Successfully loaded model from checkpoint: {checkpoint_dir}")
    return model, tokenizer


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
        else:
            raise ValueError(f"unrecognized {model.config.model_type}")
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


class ThinkPatternPostprocessWrapper(LM):
    """
    A wrapper for lm-eval models to post-process think pattern outputs.
    Removes thinking content enclosed in <think></think> tags and extracts the final answer.
    """
    def __init__(self, model: LM, start_pattern="<think>", end_pattern="</think>"):
        super().__init__()
        self.model = model
        self.start_pattern = start_pattern
        self.end_pattern = end_pattern

    def __getattr__(self, name):
        """Delegates attributes to the wrapped model."""
        return getattr(self.model, name)

    def loglikelihood(self, *args, **kwargs):
        return self.model.loglikelihood(*args, **kwargs)

    def loglikelihood_rolling(self, *args, **kwargs):
        return self.model.loglikelihood_rolling(*args, **kwargs)

    def generate_until(self, *args, **kwargs):
        """
        Calls the original model's generation and then post-processes the output
        to remove think patterns and extract final answers.
        """
        full_generations = self.model.generate_until(*args, **kwargs)
        
        processed_generations = []
        for text in full_generations:
            processed_text = self._extract_final_answer(text)
            processed_generations.append(processed_text)
            
        return processed_generations

    def _extract_final_answer(self, text):
        """
        Extract the final answer by removing think pattern content.
        """
        processed_text = text
        
        # Remove all <think>...</think> blocks
        while self.start_pattern in processed_text and self.end_pattern in processed_text:
            start_idx = processed_text.find(self.start_pattern)
            end_idx = processed_text.find(self.end_pattern, start_idx)
            
            if start_idx != -1 and end_idx != -1:
                # Remove the think block including the tags
                end_idx += len(self.end_pattern)
                processed_text = processed_text[:start_idx] + processed_text[end_idx:]
            else:
                break
        
        # Clean up extra whitespace
        processed_text = processed_text.strip()
        
        # Log the transformation for debugging
        if processed_text != text:
            logging.debug(f"Think pattern removed. Original length: {len(text)}, Processed length: {len(processed_text)}")
        
        return processed_text


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
        "log_samples": args.log_samples,
        "confirm_run_unsafe_code": True,
    }
    
    # Check if we need to load from checkpoint
    if args.checkpoint_dir:
        # Load model from checkpoint
        model, tokenizer = load_model_from_checkpoint(
            hf_model, 
            args.checkpoint_dir, 
            args.checkpoint_tag,
            args.dtype
        )
        
        # Create HFLM instance with loaded model
        hflm_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            parallelize=args.model_parallel, 
            **model_kwargs
        )
    else:
        # Original behavior - load from HF model hub
        hflm_model = HFLM(hf_model, parallelize=args.model_parallel, **model_kwargs)
    
    hflm_model.model.cuda()
    hflm_model.model.config.use_cache = True
    hflm_model.model.generation_config.use_cache = True
    if "DeepSeek-V2" in hf_model:
        hflm_model.model.generation_config.pad_token_id = hflm_model.model.generation_config.eos_token_id
    
    # This is the object that will be passed to simple_evaluate. It may be the wrapper.
    lm_obj = hflm_model
    logging.info(f"rank: {lm_obj.rank} / {lm_obj.world_size} model device: {lm_obj.model.device}, generation_config: {lm_obj.model.generation_config}, use_cache: {(lm_obj.model.generation_config.use_cache, lm_obj.model.config.use_cache)}")
    
    # Apply think pattern post-processing wrapper if enabled
    if args.postprocess_think_pattern:
        logging.info(f"Enabling think pattern post-processing with patterns: '{args.think_start_pattern}' to '{args.think_end_pattern}'")
        lm_obj = ThinkPatternPostprocessWrapper(
            hflm_model, 
            start_pattern=args.think_start_pattern,
            end_pattern=args.think_end_pattern
        )

    if args.analyse:
        # The Analyser always needs the underlying HFLM instance to attach hooks.
        analyser = Analyser(hflm_model)

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
        if args.analyse:
            analyser.dump_stats(hf_model, task, args.stats_output_dir)
            analyser.reset_stats()


if __name__ == "__main__":
    main()
