import argparse
import logging
from itertools import islice
import torch

from modepd.utils import prepare_model_and_tokenizer, register_custom_model
from modepd.dataset.nemotron_sft_dataset import build_packed_sft_dataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test script for building the packed Llama-Nemotron SFT dataset.")
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
        default="math,code",
        help="Comma-separated list of splits for training, e.g., 'math,code'.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Enable streaming for the dataset.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=4096,
        help="The block size for sequence packing.",
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

    logger.info("Starting packed SFT dataset build test...")

    # Register custom models (important for tokenizer loading if custom config is used)
    register_custom_model()

    # 1. Prepare tokenizer
    logger.info(f"Loading tokenizer from '{args.model_name_or_path}'...")
    _, tokenizer = prepare_model_and_tokenizer(args.model_name_or_path)

    # 2. Build dataset using the new packed SFT builder
    train_splits = [s.strip() for s in args.train_splits.split(',')]
    if len(train_splits) == 1:
        train_splits = train_splits[0]
        
    logger.info(f"Building packed SFT dataset '{args.dataset_name_or_path}' with config '{args.dataset_config_name}' and splits '{train_splits}'...")
    
    packed_dataset_generator = build_packed_sft_dataset(
        dataset_name=args.dataset_name_or_path,
        config_name=args.dataset_config_name,
        tokenizer=tokenizer,
        streaming=args.streaming,
        seed=42,
        split=train_splits,
        block_size=args.block_size,
    )

    logger.info("Dataset generator created. Inspecting samples...")

    # 3. Inspect a few samples from the dataset generator
    samples = list(islice(packed_dataset_generator, args.num_samples_to_show))

    if not samples:
        logger.warning("Could not retrieve any samples from the dataset generator. It might be empty or the streaming failed.")
        return
        
    for i, sample in enumerate(samples):
        logger.info(f"--- Sample {i+1} ---")
        
        # Print keys and shapes
        for key, value in sample.items():
            size = value.shape if hasattr(value, 'shape') else len(value)
            logger.info(f"  Key: '{key}', Length/Shape: {size}")

        input_ids = sample['input_ids']
        labels = sample['labels']

        # Verify that the sequence length matches the block size
        if len(input_ids) != args.block_size:
            logger.warning(f"  Sample size ({len(input_ids)}) does not match block_size ({args.block_size})!")

        # Decode and print the text to verify formatting
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        logger.info(f"  Decoded Text (first 200 chars): \n---\n{decoded_text[:200]}...\n---")

        # Verify label masking
        masked_count = (labels == -100).sum().item()
        unmasked_count = (labels != -100).sum().item()
        logger.info(f"  Label masking: {masked_count} tokens masked (prompts), {unmasked_count} tokens unmasked (responses).")
        
        # Show the transition from masked prompt to unmasked response
        try:
            first_unmasked_idx_tensor = (labels != -100).nonzero(as_tuple=True)[0]
            if len(first_unmasked_idx_tensor) > 0:
                first_unmasked_idx = first_unmasked_idx_tensor[0].item()
                
                logger.info("  Verifying a mask transition point:")
                start = max(0, first_unmasked_idx - 20)
                end = first_unmasked_idx + 20
                
                logger.info(f"    Decoded Input around transition: ...{tokenizer.decode(input_ids[start:end])}...")
                logger.info(f"    Labels around transition:        ...{labels[start:end].tolist()}...")
            else:
                logger.warning("  No unmasked tokens found in this sample.")
        except IndexError:
            logger.error("  Could not find an unmasked token to display transition.")

if __name__ == "__main__":
    main() 