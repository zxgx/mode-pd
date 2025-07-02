"""
A training script for Ulysses sequence parallelism.
It handles distributed training, checkpointing, and evaluation manually using DeepSpeed and PyTorch.
"""
import argparse
import logging 
import os
import json
from tqdm.auto import tqdm
import math
from itertools import islice

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
import transformers
from transformers import (
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from safetensors import SafetensorError

import deepspeed
from deepspeed.runtime.utils import set_random_seed, move_to_device
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter
from deepspeed.utils import groups
from deepspeed.utils.logging import logger
import deepspeed.comm as dist

from modepd.utils import register_custom_model, prepare_model_and_tokenizer, get_memory_stats
from modepd.dataset.nemotron_sft_dataset import build_packed_sft_dataset


# logger = logging.getLogger(__name__)
register_custom_model()

def parse_args():
    parser = argparse.ArgumentParser()
    # Add local_rank argument provided by deepspeed.launcher
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    
    parser.add_argument("--report_to", type=str, default="tensorboard",)
    parser.add_argument("--output_dir", type=str, default=None,)
    parser.add_argument("--evaluate_dir", type=str, default=None, help="bypass output_dir with --skip_train")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",)

    parser.add_argument("--dataset_name_or_path", type=str, default="allenai/OLMoE-mix-0924",)
    parser.add_argument("--dataset_config_name", type=str, default='SFT',)
    parser.add_argument("--train_splits", type=str, default="train", help="comma-separated list of splits for training, e.g. 'math,code'")
    parser.add_argument("--data_type", type=str, default=None)
    parser.add_argument("--streaming_dataset", action='store_true')
    parser.add_argument("--validation_dataset_name_or_path", type=str, default="Salesforce/wikitext")
    parser.add_argument("--validation_dataset_config_name", type=str, default="wikitext-2-raw-v1",)
    parser.add_argument("--validation_samples_from_train", type=int, default=None, help="Numbers of training samples to use for validation")
    parser.add_argument("--evaluate_every", type=int, default=100)

    parser.add_argument("--block_size", type=int, default=4*1024,)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,)
    parser.add_argument("--skip_train", action='store_true')
    parser.add_argument("--skip_first_batches", type=int, default=None)

    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.1,)
    parser.add_argument("--learning_rate", type=float, default=5e-5,)
    parser.add_argument("--min_lr", type=float, default=None,)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_min_lr",)
    parser.add_argument("--num_warmup_steps", type=int, default=0,)
    parser.add_argument("--max_train_steps", type=int, default=5,)
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Number of epochs to train for. Overrides max_train_steps if specified.")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None, help="Maximum steps per epoch for streaming datasets")
    parser.add_argument("--checkpointing_steps", type=int, default=-1,)
    parser.add_argument("--with_tracking", action="store_true",)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,)
    parser.add_argument("--push_to_hub", action="store_true",)
    
    parser.add_argument("--sequence_parallel_size", type=int, default=1,)

    args = parser.parse_args()
    return args


def calculate_loss(outputs, batch, model, sp_ulysses_vars):
    if sp_ulysses_vars is None:
        shift_labels = batch['labels'][..., 1:].contiguous()
        loss = outputs.loss
        total_good_tokens = sum((shift_labels != -100).view(-1))
        
    else:
        sp_group, sp_world_size = sp_ulysses_vars
        shift_labels = batch["shift_labels"]
        # model.module is the unwrapped model
        loss = model.module.loss_function(
            logits=outputs.logits,
            labels=None,
            shift_labels=shift_labels,
            vocab_size=model.module.config.vocab_size,
        )

        losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
        good_tokens = sum((shift_labels != -100).view(-1))
        good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
        total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(sp_world_size))
        total_good_tokens = sum(good_tokens_per_rank)
        
        if total_good_tokens > 0:
            loss = total_loss / total_good_tokens
        else:
            loss = torch.tensor(0.0, device=model.device)

    return loss, total_good_tokens


def calculate_training_steps(args, train_dataloader, world_size):
    """Calculate total training steps based on epochs or max_steps"""
    if args.num_train_epochs is None:
        # Use max_train_steps as before
        return args.max_train_steps, args.max_train_steps
    
    # Calculate steps per epoch
    if args.streaming_dataset or args.max_steps_per_epoch:
        # For streaming datasets, use max_steps_per_epoch or a default
        steps_per_epoch = args.max_steps_per_epoch or 1000
        logger.info(f"Using {steps_per_epoch} steps per epoch for streaming dataset")
    else:
        # For non-streaming datasets, calculate from dataset size
        try:
            dataset_size = len(train_dataloader.dataset)
            dp_world_size = world_size // args.sequence_parallel_size
            global_batch_size = args.per_device_train_batch_size * dp_world_size * args.gradient_accumulation_steps
            steps_per_epoch = math.ceil(dataset_size / global_batch_size)
            logger.info(f"Calculated {steps_per_epoch} steps per epoch from dataset size {dataset_size}")
        except (TypeError, AttributeError):
            # Fallback for datasets without __len__
            steps_per_epoch = args.max_steps_per_epoch or 1000
            logger.warning(f"Could not determine dataset size, using {steps_per_epoch} steps per epoch")
    
    total_steps = steps_per_epoch * args.num_train_epochs
    logger.info(f"Training for {args.num_train_epochs} epochs, {steps_per_epoch} steps per epoch, {total_steps} total steps")
    
    return total_steps, steps_per_epoch


def create_deepspeed_config(args):
    """Create DeepSpeed configuration with proper scheduler setup"""
    
    # calculate cosine minimum ratio if min_lr is set
    cos_min_ratio = 0.0
    if args.min_lr is not None and args.learning_rate > 0:
        cos_min_ratio = args.min_lr / args.learning_rate

    # Calculate total training steps for scheduler
    total_steps = args.max_train_steps
    if hasattr(args, '_calculated_max_steps'):
        total_steps = args._calculated_max_steps

    ds_config = {
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": { "enabled": False },
        "bf16": { "enabled": True },
        "gradient_clipping": 1.0,
        "zero_optimization": { "stage": args.zero_stage },
        "sequence_parallel_size": args.sequence_parallel_size,
    }
    
    return ds_config


def main():
    args = parse_args()
    
    # === Distributed setup ===
    deepspeed.init_distributed(dist_backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    set_random_seed(args.seed)
    world_size = dist.get_world_size()

    is_main_process = dist.get_rank() == 0
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # === Logging setup ===
    logger.setLevel(logging.INFO)
    logger.info(f"Starting script on rank {dist.get_rank()}/{world_size}")

    writer = None
    if args.with_tracking and is_main_process:
        writer = SummaryWriter(args.output_dir)

    # === Ulysses setup ===
    mpu = None
    if args.sequence_parallel_size > 1:
        mpu = UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=args.model_name_or_path,
            core_attn_implementation="sdpa",
            sequence_parallel_size=args.sequence_parallel_size,
            max_length=args.block_size,
            micro_batch_size=args.per_device_train_batch_size,
            seq_length_is_variable=True,
        )
        # Monkey-patch for deepspeed issue.
        # The bf16_optimizer expects model parallel utility functions on the mpu object,
        # but the sequence parallel mpu object does not have them.
        # We add dummy implementations for a sequence-parallel-only setup.
        if not hasattr(mpu, 'get_model_parallel_rank'):
            mpu.get_model_parallel_rank = lambda: 0
        if not hasattr(mpu, 'get_model_parallel_world_size'):
            mpu.get_model_parallel_world_size = lambda: 1
        if not hasattr(mpu, 'get_model_parallel_group'):
            # The bf16_optimizer requires a valid process group, even for a world size of 1.
            # We create a new group containing only the current process to act as a dummy.
            mpu.model_parallel_group = dist.new_group([dist.get_rank()])
            mpu.get_model_parallel_group = lambda: mpu.model_parallel_group

    # === Model and Tokenizer ===
    model, tokenizer = prepare_model_and_tokenizer(args.model_name_or_path)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # === Datasets ===
    train_dataloader = None
    validation_dataset = None
    
    if not args.skip_train:
        train_splits = [s.strip() for s in args.train_splits.split(',')]
        if len(train_splits) == 1:
            train_splits = train_splits[0]

        train_dataset = build_packed_sft_dataset(
            dataset_name=args.dataset_name_or_path,
            config_name=args.dataset_config_name,
            tokenizer=tokenizer,
            streaming=args.streaming_dataset,
            seed=args.seed,
            split=train_splits,
            block_size=args.block_size
        )
        train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size)

        # Calculate training steps based on epochs or max_steps
        max_train_steps, steps_per_epoch = calculate_training_steps(args, train_dataloader, world_size)
        args._calculated_max_steps = max_train_steps  # Store for DeepSpeed config
        
        # Override max_train_steps if using epochs
        if args.num_train_epochs is not None:
            args.max_train_steps = max_train_steps
            logger.info(f"Overriding max_train_steps to {max_train_steps} based on {args.num_train_epochs} epochs")

    if args.validation_samples_from_train:
        validation_dataset = build_packed_sft_dataset(
            dataset_name=args.dataset_name_or_path,
            config_name=args.dataset_config_name,
            tokenizer=tokenizer,
            streaming=args.streaming_dataset,
            seed=args.seed,
            split=train_splits, # Sample from the same training splits
            block_size=args.block_size,
            num_validation_samples=args.validation_samples_from_train
        )
    
    # === Optimizer and Scheduler ===
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=[0.9, 0.95])
    
    scheduler_specific_kwargs = {}
    if args.min_lr is not None:
        scheduler_specific_kwargs["min_lr"] = args.min_lr
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
        scheduler_specific_kwargs=scheduler_specific_kwargs,
    )

    # === DeepSpeed Initialization ===
    ds_config = create_deepspeed_config(args)

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
        mpu=mpu
    )

    # === Ulysses Dataloader Adapter ===
    sp_ulysses_vars = None
    if args.sequence_parallel_size > 1:
        sp_group = groups._get_sequence_parallel_group()
        sp_world_size = groups._get_sequence_parallel_world_size()
        sp_rank = groups._get_sequence_parallel_rank()
        sp_ulysses_vars = (sp_group, sp_world_size)

        if not args.skip_train:
            train_dataloader = UlyssesSPDataLoaderAdapter(train_dataloader, sp_rank, sp_group, sp_world_size, model.device)
    
    # === Checkpoint Loading ===
    completed_steps = 0
    current_epoch = 0
    if args.resume_from_checkpoint:
        _, client_state = model.load_checkpoint(args.resume_from_checkpoint)
        if client_state is not None:
            if 'step' in client_state:
                completed_steps = client_state['step']
            if 'epoch' in client_state:
                current_epoch = client_state['epoch']
            logger.info(f"Resumed from step {completed_steps}, epoch {current_epoch} from checkpoint {args.resume_from_checkpoint}")
        else:
            logger.warning(f"Could not find step in client state from checkpoint {args.resume_from_checkpoint}")

    # === Training Loop ===
    if not args.skip_train:
        logger.info("***** Running training *****")
        if args.num_train_epochs is not None:
            logger.info(f"Training for {args.num_train_epochs} epochs, starting from epoch {current_epoch}")
            logger.info(f"Steps per epoch: {steps_per_epoch}")
        else:
            logger.info(f"Training for {args.max_train_steps} steps")
        
        progress_bar = tqdm(range(completed_steps, args.max_train_steps), disable=not is_main_process)
        model.train()

        accumulation_step = 0
        any_good_tokens_in_accumulation = False
        
        # Multi-epoch training loop
        for epoch in range(current_epoch, args.num_train_epochs if args.num_train_epochs else current_epoch + 1):
            if args.num_train_epochs is not None:
                logger.info(f"Starting epoch {epoch + 1}/{args.num_train_epochs}")
            
            active_dataloader = train_dataloader
            
            # Handle resuming within an epoch
            if epoch == current_epoch and completed_steps > 0:
                batches_to_skip = (completed_steps - epoch * steps_per_epoch) * args.gradient_accumulation_steps
                if batches_to_skip > 0:
                    logger.info(f"Skipping {batches_to_skip} batches to resume training within epoch {epoch + 1}")
                    active_dataloader = islice(train_dataloader, batches_to_skip, None)
            
            epoch_step = 0
            for batch_idx, batch in enumerate(active_dataloader):
                batch = move_to_device(batch, device)
                outputs = model(**batch)
                loss, num_good_tokens = calculate_loss(outputs, batch, model, sp_ulysses_vars)
                
                if num_good_tokens > 0:
                    any_good_tokens_in_accumulation = True

                # Scale loss by gradient accumulation steps
                loss = loss / args.gradient_accumulation_steps
                
                model.backward(loss)
                accumulation_step += 1
                
                # Only step and zero grad every gradient_accumulation_steps
                if accumulation_step % args.gradient_accumulation_steps == 0:
                    if any_good_tokens_in_accumulation:
                        model.step()
                    else:
                        # We are skipping the step, but still need to zero the gradients
                        logger.info(f"Skipping optimizer step at step {completed_steps} because no good tokens were seen in this accumulation period.")
                    
                    model.optimizer.zero_grad()
                    
                    # Reset for the next accumulation window and update progress
                    any_good_tokens_in_accumulation = False
                    
                    progress_bar.update(1)
                    completed_steps += 1
                    epoch_step += 1
                    accumulation_step = 0
                    
                    if writer:
                        # Log the unscaled loss for monitoring
                        writer.add_scalar("loss/train", (loss * args.gradient_accumulation_steps).item(), completed_steps)
                        writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], completed_steps)
                        writer.add_scalar("epoch", epoch + (epoch_step / steps_per_epoch), completed_steps)
                        if sp_ulysses_vars is None:
                            writer.add_scalar("good_tokens/train", num_good_tokens.item(), completed_steps)

                    if args.checkpointing_steps > 0 and completed_steps % args.checkpointing_steps == 0:
                        ckpt_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                        model.save_checkpoint(ckpt_dir, client_state={'step': completed_steps, 'epoch': epoch})

                    if completed_steps % args.evaluate_every == 0:
                        if validation_dataset is not None:
                            evaluate(args, model, validation_dataset, sp_ulysses_vars, completed_steps, writer, is_main_process)
                        else:
                            logger.info(f"Skipping evaluation at step {completed_steps} as no validation set is configured.")
                        model.train()

                    if completed_steps >= args.max_train_steps:
                        break
                
                # For streaming datasets, break after max_steps_per_epoch if specified
                if args.streaming_dataset and args.max_steps_per_epoch and epoch_step >= args.max_steps_per_epoch:
                    logger.info(f"Reached max_steps_per_epoch ({args.max_steps_per_epoch}) for epoch {epoch + 1}")
                    break
            
            if args.num_train_epochs is not None:
                logger.info(f"Completed epoch {epoch + 1}/{args.num_train_epochs}, steps: {epoch_step}")
            
            if completed_steps >= args.max_train_steps:
                break
    
    # === Final Evaluation and Saving ===
    logger.info("***** Running Final Evaluation *****")
    if validation_dataset is not None:
        evaluate(args, model, validation_dataset, sp_ulysses_vars, completed_steps, writer, is_main_process)

    if args.output_dir is not None and is_main_process:
        logger.info(f"Saving final model to {args.output_dir}")
        model.module.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if writer:
        writer.close()
    
    logger.info("Training finished.")


def evaluate(args, model, validation_dataset, sp_ulysses_vars, completed_steps, writer, is_main_process):
    model.eval()
    losses = []

    validation_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_train_batch_size)
    if sp_ulysses_vars is not None:
        sp_group, sp_world_size = sp_ulysses_vars
        sp_rank = groups._get_sequence_parallel_rank()
        validation_dataloader = UlyssesSPDataLoaderAdapter(validation_dataloader, sp_rank, sp_group, sp_world_size, model.device)
    
    num_eval_steps = None
    if args.validation_samples_from_train:
        world_size = dist.get_world_size()
        dp_world_size = world_size // args.sequence_parallel_size
        global_batch_size = args.per_device_train_batch_size * dp_world_size
        # The Ulysses adapter will handle the sharding across SP ranks.
        # We need to calculate the number of steps for the data parallel dimension.
        num_eval_steps = math.ceil(args.validation_samples_from_train / global_batch_size)

    eval_iterator = tqdm(validation_dataloader, total=num_eval_steps, desc=f"Evaluation at step {completed_steps}", disable=not is_main_process)
    for batch in eval_iterator:
        batch = move_to_device(batch, model.device)
        with torch.no_grad():
            outputs = model(**batch)
            loss, _ = calculate_loss(outputs, batch, model, sp_ulysses_vars)
        
        # Gather losses from all ranks
        world_size = dist.get_world_size()
        gathered_losses = [torch.zeros_like(loss) for _ in range(world_size)]
        dist.all_gather(gathered_losses, loss)
        losses.extend(gathered_losses)

    # Since all ranks have all losses, we only need to calculate perplexity on the main process
    if is_main_process:
        if not losses:
            logger.warning(f"Evaluation at step {completed_steps} resulted in no losses, skipping perplexity calculation.")
            return

        try:
            eval_loss = torch.mean(torch.stack(losses))
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float('inf')
        
        logger.info(f"step: {completed_steps} perplexity: {perplexity} eval_loss: {eval_loss.item()}")
        if writer:
            writer.add_scalar("perplexity", perplexity, completed_steps)
            writer.add_scalar("loss/eval", eval_loss.item(), completed_steps)


if __name__ == "__main__":
    main()