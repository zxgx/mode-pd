import argparse
import os
from pathlib import Path
import json
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--speedup", action="store_true")
    parser.add_argument("--math", action="store_true")

    return parser.parse_args()


def speedup(args):
    seed_list = [42, 1020, 1998]
    model_list = ["origin", "q3-50", "q3-75"]
    data_dict = {
        "seed": [],
        "model": [],
        "batch_size": [],
        "input_len": [],
        "output_len": [],
        "novice_hit_ratios": [],
        "total_latency": [],
        "speedup": [],
    }
    for batch_size in [1, 32, 128, 512]:
        for output_len in [128, 256, 512]:
            for seed in seed_list:
                base_latency = None
                for model in model_list:
                    path = os.path.join(args.log_dir, f"benchmark-s{seed}-{model}.jsonl")
            
                    with open(path) as f:
                        for line in f:
                            src = json.loads(line)
                            if src["batch_size"] == batch_size and src["output_len"] == output_len:                            
                                data_dict["seed"].append(seed)
                                data_dict["model"].append(model)
                                data_dict["batch_size"].append(src["batch_size"])
                                data_dict["input_len"].append(src["input_len"])
                                data_dict["output_len"].append(src["output_len"])
                                hit_ratio = src.get("novice_hit_ratios", 0)
                                data_dict["novice_hit_ratios"].append(hit_ratio)
                                data_dict["total_latency"].append(src["total_latency"])
                                if base_latency is None:
                                    base_latency = src["total_latency"]
                                data_dict["speedup"].append(base_latency / src["total_latency"])


    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(args.log_dir, f"benchmark-summary.csv"), index=False)


def performace(args):
    data_dict = {
        "arc_challenge": {},
        "arc_easy": {},
        "boolq": {},
        "copa": {},
        "mmlu": {},
        "openbookqa": {},
        "piqa": {},
        "rte": {},
        "winogrande": {},
    }
    task_keys = [
        "ai2_arc", "boolq", "copa", "mmlu", 
        "openbookqa", "piqa", "rte", "winogrande"
    ]
    model_set = set()

    directory = Path(args.log_dir)
    for entry in directory.iterdir():
        if entry.is_dir() or not entry.name.endswith(".json"):
            continue
        base = entry.name.split('-')
        model_id = '-'.join(base[:-1])
        model_set.add(model_id)

    model_list = sorted(model_set)
    for task in task_keys:
        for model_id in model_list:
            log_path = os.path.join(args.log_dir, f"{model_id}-{task}.json")
            print(f"loadijng: {log_path}")
            with open(log_path) as f:
                src = json.load(f)

            if task == "ai2_arc":
                data_dict['arc_challenge'][model_id] = src['results']['arc_challenge']['acc_norm,none']
                data_dict['arc_easy'][model_id] = src['results']['arc_easy']['acc_norm,none']
            elif task == "boolq":
                data_dict[task][model_id] = src['results'][task]['acc,none']
            elif task == "copa":
                data_dict[task][model_id] = src['results'][task]['acc,none']
            elif task == "mmlu":
                data_dict[task][model_id] = src['results'][task]['acc,none']
            elif task == "openbookqa":
                data_dict[task][model_id] = src['results'][task]['acc_norm,none']
            elif task == "piqa":
                data_dict[task][model_id] = src['results'][task]['acc_norm,none']
            elif task == "rte":
                data_dict[task][model_id] = src['results'][task]['acc,none']
            elif task == "winogrande":
                data_dict[task][model_id] = src['results'][task]['acc,none']
    
    df = pd.DataFrame(data_dict) * 100
    df['average'] = df.mean(axis=1)
    df.to_csv(os.path.join(args.log_dir, f"summary-{args.log_dir.replace('/', '_')}.csv"))


def math_perf(args):
    data_dict = {
        "gsm8k": {},
        # "hendrycks_math": {},
        "minerva_math": {},
    }
    task_keys = [
        "gsm8k", "minerva_math"
    ]
    model_set = set()

    directory = Path(args.log_dir)
    for entry in directory.iterdir():
        if entry.is_dir() or not entry.name.endswith(".json"):
            continue
        base = entry.name.split('-')
        model_id = '-'.join(base[:-1])
        model_set.add(model_id)

    model_list = sorted(model_set)
    for task in task_keys:
        for model_id in model_list:
            log_path = os.path.join(args.log_dir, f"{model_id}-{task}.json")
            print(f"loading: {log_path}")
            with open(log_path) as f:
                src = json.load(f)

            if task == "gsm8k":
                data_dict[task][model_id] = src['results'][task]['exact_match,flexible-extract']
            elif task == "hendrycks_math":
                data_dict[task][model_id] = src['results'][task]['exact_match,none']
            elif task == "minerva_math":
                data_dict[task][model_id] = src['results'][task]['exact_match,none']
    
    df = pd.DataFrame(data_dict) * 100
    # df['average'] = df.mean(axis=1)
    df.to_csv(os.path.join(args.log_dir, f"summary-{args.log_dir.replace('/', '_')}.csv"))


def main():
    args = get_args()

    if args.speedup:
        speedup(args)
    elif args.math:
        math_perf(args)
    else:
        performace(args)


if __name__ == "__main__":
    main()
