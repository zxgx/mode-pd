import argparse
import logging
from itertools import islice

from modepd.utils import build_dataset, prepare_model_and_tokenizer, register_custom_model

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test script for building the Llama-Nemotron dataset.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path to load the tokenizer from.",
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="nvidia/Llama-Nemotron-Post-Training-Dataset",
        help="Name or path of the dataset to load.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="SFT",
        help="Configuration name for the dataset.",
    )
    parser.add_argument(
        "--train_splits",
        type=str,
        default="math",
        help="Comma-separated list of splits for training, e.g., 'math,code'.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming for the dataset.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=4096,
        help="The block size for tokenization and grouping texts.",
    )
    parser.add_argument(
        "--num_samples_to_show",
        type=int,
        default=3,
        help="Number of processed samples to display.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    logger.info("Starting dataset build test...")

    # Register custom models (important for tokenizer loading if custom config is used)
    register_custom_model()

    # 1. Prepare tokenizer
    logger.info(f"Loading tokenizer from '{args.model_name_or_path}'...")
    # We only need the tokenizer for this test
    _, tokenizer = prepare_model_and_tokenizer(args.model_name_or_path)

    # 2. Build dataset
    train_splits = [s.strip() for s in args.train_splits.split(',')]
    if len(train_splits) == 1:
        train_splits = train_splits[0]
        
    logger.info(f"Building dataset '{args.dataset_name_or_path}' with config '{args.dataset_config_name}' and splits '{train_splits}'...")
    
    # We set `is_validation=False` to test the main training data processing path (with grouping)
    # Set accelerator=None as we are running on a single process
    lm_dataset = build_dataset(
        dataset_name_or_path=args.dataset_name_or_path,
        dataset_config_name=args.dataset_config_name,
        streaming=args.streaming,
        tokenizer=tokenizer,
        split=train_splits,
        block_size=args.block_size,
        logger=logger,
        accelerator=None, # Not using accelerator for this test
        seed=42,
        is_validation=False,
    )

    logger.info("Dataset built successfully. Inspecting samples...")

    # 3. Inspect a few samples from the dataset
    if args.streaming:
        samples = list(islice(lm_dataset, args.num_samples_to_show))
    else:
        samples = lm_dataset.select(range(args.num_samples_to_show))

    if not samples:
        logger.warning("Could not retrieve any samples from the dataset. It might be empty or the streaming failed.")
        return
        
    for i, sample in enumerate(samples):
        logger.info(f"--- Sample {i+1} ---")
        
        # Print keys and shapes
        for key, value in sample.items():
            # Tensors have shape, lists have len
            size = value.shape if hasattr(value, 'shape') else len(value)
            logger.info(f"  Key: '{key}', Length/Shape: {size}")

        # Decode and print the text to verify formatting
        input_ids = sample['input_ids']
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        
        logger.info(f"  Decoded Text (first 200 chars): \n---\n{decoded_text[:200]}...\n---")
        logger.info(f"  Decoded Text (last 200 chars): \n---\n...{decoded_text[-200:]}\n---")


if __name__ == "__main__":
    main() 