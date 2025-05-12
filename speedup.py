import argparse
import time
import json
import numpy as np
from tqdm import tqdm
import os

import torch

from modepd.utils import register_custom_model, prepare_model_and_tokenizer


register_custom_model()


def _handle_non_serializable(o):
    """Copied from https://github.com/meta-llama/llama-recipes/blob/b5f64c0b69d7ff85ec186d964c6c557d55025969/tools/benchmarks/llm_eval_harness/eval.py#L18
    """
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-V2-Lite-Chat",)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    model, tokenizer = prepare_model_and_tokenizer(args.model_name_or_path)
    model.cuda()
    model.eval()
    model.config.use_cache = True
    model.generation_config.use_cache = True

    # warm up
    text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    for _ in range(5):
        outputs = model(**inputs)
    
    task_keys = [
        "arc_challenge", "arc_easy", "boolq", "copa", "mmlu", 
        "openbookqa", "piqa", "rte", "winogrande"
    ]

    results = {}
    total_throughput = []
    for task in task_keys:
        if task == 'arc_challenge' or task == 'arc_easy':
            path = f"exp/samples/_mnt_videodata_zhgeng_models_Qwen2.5-0.5B-ai2_arc.json"
        else:
            path = f"exp/samples/_mnt_videodata_zhgeng_models_Qwen2.5-0.5B-{task}.json"
        throughput_list = []
        with open(path) as f:
            data = json.load(f)
        if task == "mmlu":
            for key in data['samples']:
                for idx, sample in tqdm(enumerate(data['samples'][key])):
                    if args.limit is not None and idx == args.limit:
                        break
                    input_text = sample["arguments"][0][0]
                    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                    prefill_len = inputs['input_ids'].shape[1]

                    torch.cuda.synchronize()
                    start = time.time()

                    outputs = model(**inputs)

                    torch.cuda.synchronize()
                    end = time.time()

                    prefill_time = end - start
                    throughput = prefill_len / prefill_time
                    throughput_list.append(throughput)
                    total_throughput.append(throughput)
        else:
            for idx, sample in tqdm(enumerate(data['samples'][task])):
                if args.limit is not None and idx == args.limit:
                    break
                input_text = sample["arguments"][0][0]
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                prefill_len = inputs['input_ids'].shape[1]

                torch.cuda.synchronize()
                start = time.time()

                outputs = model(**inputs)

                torch.cuda.synchronize()
                end = time.time()

                prefill_time = end - start
                throughput = prefill_len / prefill_time
                throughput_list.append(throughput)
                total_throughput.append(throughput)
        # print(f"prefill_len: {prefill_len}, prefill time: {prefill_time:.2f}, throughput: {prefill_len / prefill_time:.2f}")

        results[task] = {
            "mean": np.mean(throughput_list),
            "std": np.std(throughput_list)
        }
        print(f"throughput for task: {task}: {np.mean(throughput_list):.2f} +- {np.std(throughput_list):.2f}")
    
    results['total'] = {
        "mean": np.mean(total_throughput),
        "std": np.std(total_throughput)
    }
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"{args.model_name_or_path.replace('/', '_')}.json"), "w") as f:
                json.dump(results, f, default=_handle_non_serializable, indent=2)


if __name__ == "__main__":

    main()
    