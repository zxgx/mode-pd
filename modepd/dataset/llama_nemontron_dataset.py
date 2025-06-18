from datasets import load_dataset, concatenate_datasets
import logging
import json

logger = logging.getLogger(__name__)

def load_llama_nemotron_dataset(name_or_path, config_name=None, streaming=False, seed=None, split='train'):
    """
    Loads the Llama-Nemotron dataset from NVIDIA.
    This function specifically handles the 'SFT' configuration of the dataset,
    which has 'input' and 'output' columns for supervised fine-tuning.
    The 'input' column is a JSON string or list of dicts representing the conversation history.
    """
    logger.info(f"Loading Llama-Nemotron dataset '{name_or_path}' with config '{config_name}' and split '{split}'")
    # The split can be a list of strings for this dataset, e.g., ["math", "code"]
    dataset = load_dataset(name_or_path, name=config_name, split=split, streaming=streaming)
    
    if isinstance(dataset, list):
        dataset = concatenate_datasets(dataset)

    def format_sft_example(example):
        # The 'input' can be a string representing a list of dicts.
        try:
            # The input can be a string that needs to be parsed as JSON.
            if isinstance(example['input'], str):
                messages = json.loads(example['input'])
            else:
                messages = example['input']
        except (json.JSONDecodeError, TypeError):
            # If input is not a valid JSON string or not a list, treat it as a simple prompt.
            messages = [{"role": "user", "content": str(example['input'])}]

        # Format chat history. Using a simple template.
        prompt_str = ""
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and 'role' in message and 'content' in message:
                    prompt_str += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        
        response_str = example.get('output', '')
        
        # Combine prompt and response for training.
        example['text'] = prompt_str + f"<|im_start|>assistant\n{response_str}<|im_end|>"
        return example

    if config_name == "SFT":
        # The mapping will add a 'text' column.
        # The original columns will be removed later in build_dataset.
        dataset = dataset.map(format_sft_example)
    else:
        # Fallback for other configurations, assuming 'prompt' and 'response' columns.
        def combine_columns(example):
            example['text'] = example['prompt'] + ' ' + example['response']
            return example
        dataset = dataset.map(combine_columns)

    return dataset
