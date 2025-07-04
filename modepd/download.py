from huggingface_hub import snapshot_download
# from datasets import load_dataset

# zh = load_dataset("allenai/c4", "zh", streaming=True)

models = [
    # "deepseek-ai/DeepSeek-V2.5-1210", 
    # 'deepseek-ai/DeepSeek-V2-Lite', 
    # 'deepseek-ai/DeepSeek-V2-Lite-Chat', 
    'Qwen/Qwen2.5-3B', 
    'Qwen/Qwen2.5-3B-Instruct',
    'google/gemma-2-2b-it',
    'google/gemma-2-2b',
    'moonshotai/Moonlight-16B-A3B',
    'moonshotai/Moonlight-16B-A3B-Instruct',
]
for model in models:
    snapshot_download(model)
