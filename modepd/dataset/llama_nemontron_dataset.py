from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def load_llama_nemotron_dataset(name_or_path, config_name=None, streaming=False, seed=None, split='train'):
    """
    Loads the Llama-Nemotron dataset.
    This dataset has 'prompt' and 'response' columns. We combine them into a 'text' column.
    """
    logger.info(f"Loading Llama-Nemotron dataset '{name_or_path}' with config '{config_name}' and split '{split}'")
    dataset = load_dataset(name_or_path, name=config_name, split=split, streaming=streaming)

    def combine_columns(example):
        # Combining prompt and response to form a single text sequence for language modeling.
        # A space is added for clear separation.
        example['text'] = example['prompt'] + ' ' + example['response']
        return example

    # The mapping will add a 'text' column. The original columns will be removed later in build_dataset.
    dataset = dataset.map(combine_columns)

    return dataset
