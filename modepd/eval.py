import os
import logging
import argparse
from pprint import pformat
import json
import numpy as np
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM

from modepd.model.configuration_deepseek import DeepseekV2Config
from modepd.model.modeling_deepseek import DeepseekV2Model, DeepseekV2ForCausalLM

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

AutoConfig.register("deepseek_v2_compressed", DeepseekV2Config)
AutoModel.register(DeepseekV2Config, DeepseekV2Model)
AutoModelForCausalLM.register(DeepseekV2Config, DeepseekV2ForCausalLM)

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
    # lm_eval config
    parser.add_argument(
        "--tasks", type=str, nargs='+',
        default=[
            # English
            "mmlu", 
            # Math
            "gsm8k",
            # Code
            "humaneval",
            # Chinese
            "cmmlu",
            ])
    parser.add_argument(
        "--num_fewshots", type=int, nargs='+',
        default=[
            # English
            # "mmlu",
            5,
            # Math
            # "gsm8k",
            8,
            # Code
            # "humaneval"
            0,
            # Chinese
            # "cmmlu",
            5,
            ])
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()


def main():
    logging.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', None)}")
    args = get_args()
    logging.info(f"{pformat(vars(args), indent=2, width=120)}")

    hf_model = args.hf_model
    
    model_kwargs = {
        "trust_remote_code": True if args.trust_remote_code else None,
        # "torch_dtype": args.dtype,  # override by `dtype`` in lm_eval.HFLM
        "dtype": args.dtype,
        "device_map": "auto",
        "batch_size": args.batch_size, #"auto:4",
        "backend": "causal"
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
    
    lm_eval_kwargs = {
        "limit": args.limit,
        "log_samples": False,
        "confirm_run_unsafe_code": True,
    }
    
    lm_obj = HFLM(hf_model, parallelize=True, **model_kwargs)
    if "DeepSeek-V2" in hf_model:
        lm_obj.model.config.use_cache = True
        lm_obj.model.generation_config.use_cache = True
        lm_obj.model.generation_config.pad_token_id = lm_obj.model.generation_config.eos_token_id
    print(f"model device: {lm_obj.model.device}, generation_config: {lm_obj.model.generation_config}, use_cache: {(lm_obj.model.generation_config.use_cache, lm_obj.model.config.use_cache)}")
    
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


if __name__ == "__main__":
    main()
