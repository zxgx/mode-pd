import argparse
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-V2-Lite-Chat",)

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, use_cache=False)
    
    import pdb; pdb.set_trace()

    if "DeepSeek-V2" in model_name:
        model.generation_config = GenerationConfig.from_pretrained(model_name)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id


if __name__ == "__main__":
    main()
    