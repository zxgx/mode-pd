import argparse
import os
from pathlib import Path
import json
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--log_dir", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()

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


if __name__ == "__main__":
    main()
