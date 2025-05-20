"""
Reference: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
"""
import argparse
import logging
import os
import json
from tqdm.auto import tqdm
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import datasets
import transformers
from transformers import (
    DataCollatorForTokenClassification,
    get_scheduler,
)
from safetensors import SafetensorError

from modepd.utils import register_custom_model, prepare_model_and_tokenizer, get_memory_stats, build_dataset
from modepd.dataset.sft_dataset import load_sft_dataset


logger = get_logger(__name__)
register_custom_model()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_tracking", action="store_true",)
    parser.add_argument("--report_to", type=str, default="tensorboard",)
    parser.add_argument("--output_dir", type=str, default=None,)
    parser.add_argument("--evaluate_dir", type=str, default=None, help="bypass output_dir with --skip_train")
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",)

    parser.add_argument("--dataset_name_or_path", type=str, default="allenai/OLMoE-mix-0924",)
    parser.add_argument("--data_type", type=str, default=None)
    parser.add_argument("--evaluate_every", type=int, default=100)

    parser.add_argument("--block_size", type=int, default=4*1024,)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,)
    parser.add_argument("--skip_train", action='store_true')
    parser.add_argument("--skip_first_batches", type=int, default=None)

    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.1,)
    parser.add_argument("--learning_rate", type=float, default=1e-5,)
    parser.add_argument("--min_lr", type=float, default=None,)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_min_lr",)
    parser.add_argument("--num_warmup_steps", type=int, default=0,)
    parser.add_argument("--warmup_ratio", type=float, default=None,)
    parser.add_argument("--max_train_steps", type=int, default=None,)
    parser.add_argument("--checkpointing_steps", type=int, default=-1,)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,)
    parser.add_argument("--push_to_hub", action="store_true",)

    parser.add_argument("--distillation", action='store_true')
    parser.add_argument("--teacher_model_name_or_path", type=str, default=None)
    parser.add_argument("--sft_cache_path", type=str, default=None)
    parser.add_argument("--validation_split_percentage", type=float, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--distillation_alpha", type=float, default=0.5,
        help="Weight for distillation loss (0 to 1). Higher values put more weight on matching teacher outputs."
    )
    parser.add_argument(
        "--distillation_temperature", type=float, default=2.0,
        help="Temperature for softening probability distributions in distillation."
    )
    parser.add_argument("--disable_batch_aggregation", action='store_true')
    return parser.parse_args()


def calculate_loss(outputs, batch, model, teacher_model, distillation_temperature, distillation_alpha, batch_aggregation=True):
    # Shift so that tokens < n predict n
    lm_loss = outputs.loss
    shift_labels = batch['labels'][..., 1:].contiguous()
    mask = shift_labels != -100
    num_valid_tokens = mask.sum()

    if teacher_model is not None:        
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        with torch.no_grad():
            teacher_outputs = teacher_model(**batch)
            shift_teacher_logits = teacher_outputs.logits[..., :-1, :].contiguous()

        masked_student_logits = shift_logits[mask] / distillation_temperature
        masked_teacher_logits = shift_teacher_logits[mask] / distillation_temperature

        distill_loss = F.kl_div(
            input=F.log_softmax(masked_student_logits, dim=-1),
            target=F.softmax(masked_teacher_logits, dim=-1),
            reduction='batchmean',
            log_target=False,
        )

        distill_loss = distill_loss * (distillation_temperature**2)
        
        loss = (1-distillation_alpha) * lm_loss + distillation_alpha * distill_loss
    else:
        loss = lm_loss
        distill_loss = None

    if batch_aggregation:
        loss *= num_valid_tokens.float()
    
    return loss, distill_loss


def main():
    args = parse_args()

    #################
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    deepspeed_plugin = DeepSpeedPlugin(
        gradient_clipping=1.0,
        zero_stage=args.zero_stage,
        zero3_save_16bit_model=True,
    )
    if args.distillation:
        inference_plugin = DeepSpeedPlugin(
            hf_ds_config={
                "bf16": {
                    "enabled": True
                },
                "zero_optimization": {
                    "stage": 0,
                    "overlap_comm": True,
                },
                "train_micro_batch_size_per_gpu": 1
            }
        )
        deepspeed_plugins = {"student": deepspeed_plugin, "teacher": inference_plugin}
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="bf16",
            deepspeed_plugins=deepspeed_plugins,
            **accelerator_log_kwargs
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            deepspeed_plugin=deepspeed_plugin,
            mixed_precision="bf16",
            **accelerator_log_kwargs
        )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"{accelerator.state}\n rank: {accelerator.process_index}/{accelerator.num_processes}: "
                f"is_main_process: {accelerator.is_main_process}, is_local_main_process: {accelerator.is_local_main_process}", 
                main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # TODO: Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    #################
    # Prepare model & tokenizer
    model, tokenizer = prepare_model_and_tokenizer(args.model_name_or_path)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: {trainable_params/1024**3:.2f} B, device: {model.device}, dtype: {model.dtype}, "
        f"emb size: {embedding_size}, tokenizer vocab size: {len(tokenizer)}")

    teacher_model = None
    if args.distillation:
        teacher_model, _ = prepare_model_and_tokenizer(args.teacher_model_name_or_path)
        assert embedding_size == teacher_model.get_input_embeddings().weight.shape[0]

        teacher_model.eval()
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        logger.info(f"Teacher model parameters: {teacher_params/1024**3:.2f} B, device: {teacher_model.device}, dtype: {teacher_model.dtype}")

    alloc, max_alloc, reserved, max_reserved = get_memory_stats()
    logger.info(
        f"Memory stats after initializing model: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {reserved:.2f} G / {max_reserved:.2f} G", 
        main_process_only=False)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        if teacher_model:
            teacher_model.resize_token_embeddings(len(tokenizer))
    
    # 3. DataLoaders creation
    # Load training dataset
    full_train_dataset = load_sft_dataset(
        args.dataset_name_or_path, tokenizer, 'train', data_type=args.data_type,
        block_size=args.block_size, logger=logger, accelerator=accelerator,
        seed=args.seed, cache_path=args.sft_cache_path
    )['train']

    train_val_split_point = int(len(full_train_dataset) * (100 - args.validation_split_percentage) / 100)
    # Create train/val split
    train_dataset = full_train_dataset.select(range(train_val_split_point))
    validation_dataset = full_train_dataset.select(range(train_val_split_point, len(full_train_dataset)))
    
    logger.info(f"Split dataset: {len(train_dataset)} training examples, {len(validation_dataset)} validation examples")

    # Create DataLoaders
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    
    validation_dataloader = DataLoader(
        validation_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    #################
    # Prepare optimizer and scheduler
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, betas=[0.9, 0.95], weight_decay=args.weight_decay)
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    if args.warmup_ratio is not None:
        warmup_steps = int(args.max_train_steps * args.warmup_ratio)
    else:
        warmup_steps = args.num_warmup_steps * accelerator.num_processes
    scheduler_specific_kwargs = {}
    if args.min_lr is not None:
        scheduler_specific_kwargs["min_lr"] = args.min_lr
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes,
        scheduler_specific_kwargs=scheduler_specific_kwargs,
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
    )
    if args.distillation:
        accelerator.state.select_deepspeed_plugin("teacher")
        teacher_model = accelerator.prepare(teacher_model)
        accelerator.state.select_deepspeed_plugin("student")
        teacher_model.eval()
    logger.info(f"optimizer type: {type(optimizer)}", main_process_only=False)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    alloc, max_alloc, reserved, max_reserved = get_memory_stats()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"{model}")
    logger.info(
        f"Model parameters: {trainable_params/1024**3:.2f} B, device: {model.device}, dtype: {model.dtype}"
        f", Memory stats before training: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {reserved:.2f} G / {max_reserved:.2f} G"
        , main_process_only=False)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        accelerator.init_trackers("sft-training", experiment_config)

    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        dirs = [f.path for f in os.scandir(args.resume_from_checkpoint) if f.is_dir() and f.name.startswith("step_")]
        if len(dirs) > 0:
            dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
            checkpoint_path = dirs[-1]
            path = os.path.basename(checkpoint_path)

            accelerator.load_state(os.path.join(checkpoint_path, "ckpt"))
            # Extract `step_{i}`
            training_difference = os.path.splitext(path)[0]

            # need to multiply `gradient_accumulation_steps` to reflect real steps
            completed_steps = int(training_difference.replace("step_", ""))
            resume_step = completed_steps * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            logger.info(
                f"Resumed from checkpoint: {checkpoint_path}, completed steps: {completed_steps}, w. grad acc: {completed_steps * args.gradient_accumulation_steps}, "
                f"starting epoch: {starting_epoch}, step: {resume_step}"
            )
        else:
            resume_step = None
            logger.warning(
                f"Please be aware that resume_from_checkpoint is specified as {args.resume_from_checkpoint}, "
                f"but no ckpt is detected"
            )

    #################
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            step_loss = torch.zeros(1, device=model.device, dtype=torch.float)
            if args.distillation:
                step_distill_loss = torch.zeros(1, device=model.device, dtype=torch.float)
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                # loss = outputs.loss
                loss, distill_loss = calculate_loss(
                    outputs, batch, model, teacher_model, args.distillation_temperature, 
                    args.distillation_alpha, not args.disable_batch_aggregation
                )

                if args.with_tracking:
                    step_loss += loss.detach().float()
                    if args.distillation:
                        step_distill_loss += distill_loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if args.with_tracking:
                    step_loss /= args.gradient_accumulation_steps
                    global_loss = accelerator.reduce(step_loss, reduction='mean')
                    log_info = {"train_loss": global_loss.item(),}
                    if args.distillation:
                        step_distill_loss /= args.gradient_accumulation_steps
                        global_distill_loss = accelerator.reduce(step_distill_loss, reduction='mean')
                        log_info = {"distill_loss": global_distill_loss.item()}
                    for lr_idx, lr in enumerate(lr_scheduler.get_last_lr()):
                        log_info[f"lr_{lr_idx}"] = lr
                    
                    accelerator.log(log_info, step=completed_steps)
                    step_loss.zero_()

                if args.checkpointing_steps > 0 and completed_steps % args.checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    # accelerator.save_state(os.path.join(output_dir, "ckpt"))

                    try:
                        model_dir = os.path.join(output_dir, "model")
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            model_dir, is_main_process=accelerator.is_main_process, 
                            save_function=accelerator.save, state_dict=accelerator.get_state_dict(model)
                        )
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(model_dir)
                        logger.info(f"model&ckpt is saved to {output_dir}", main_process_only=False)
                    except SafetensorError as e:
                        logger.warning(f"rank {accelerator.process_index} at step {step} fails in saving model.", main_process_only=False)
                    accelerator.wait_for_everyone()

                if completed_steps % args.evaluate_every == 0:
                    losses = []
                    model.eval()
                    for step, batch in tqdm(
                        enumerate(validation_dataloader), total=len(validation_dataloader), desc=f"Evaluation at step {completed_steps}",
                        disable=not accelerator.is_local_main_process
                    ):
                        with torch.no_grad():
                            outputs = model(**batch)
                        loss = outputs.loss.repeat(args.per_device_train_batch_size)
                        losses.append(accelerator.gather(loss))

                    losses = torch.cat(losses)
                    try:
                        valid_loss = torch.mean(losses)
                        perplexity = math.exp(valid_loss)
                    except OverflowError:
                        perplexity = float('inf')
                    
                    logger.info(f"step: {completed_steps} perplexity: {perplexity} eval_loss: {valid_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "perplexity": perplexity,
                                "eval_loss": valid_loss,
                            },
                            step=completed_steps
                        )
                    model.train()
                            
            if completed_steps >= args.max_train_steps:
                break
    
    losses = []
    model.eval()
    for step, batch in tqdm(
        enumerate(validation_dataloader), total=len(validation_dataloader), desc=f"Evaluation at step {completed_steps}",
        disable=not accelerator.is_local_main_process
    ):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss.repeat(args.per_device_train_batch_size)
        losses.append(accelerator.gather(loss))

    losses = torch.cat(losses)
    try:
        valid_loss = torch.mean(losses)
        perplexity = math.exp(valid_loss)
    except OverflowError:
        perplexity = float('inf')
    
    logger.info(f"step: {completed_steps} perplexity: {perplexity} eval_loss: {valid_loss}")
    if args.with_tracking:
        accelerator.log(
            {
                "perplexity": perplexity,
                "eval_loss": valid_loss,
            },
            step=completed_steps
        )

    if args.evaluate_dir is not None:
        if accelerator.is_main_process:
            with open(os.path.join(args.evaluate_dir, f"{args.model_name_or_path.replace('/', '_')}-perplexity.json"), "w") as f:
                json.dump({
                    "step": completed_steps, "eval_loss": valid_loss.item(), "perplexity": perplexity,
                }, f)

    alloc, max_allc, resv, max_resv = get_memory_stats()
    logger.info(
        f"Memory stats on exiting: Alloc: {alloc:.2f} G / {max_allc:.2f} G, Resrv: {resv:.2f} G / {max_resv:.2f} G"
        , main_process_only=False)

    if args.output_dir is not None:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, 
            save_function=accelerator.save, state_dict=accelerator.get_state_dict(model)
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            with open(os.path.join(args.output_dir, f"perplexity.json"), "w") as f:
                json.dump({"step": completed_steps, "eval_loss": valid_loss.item(), "perplexity": perplexity,}, f)
        logger.info(f"Saving model to {args.output_dir} done!", main_process_only=False)
        accelerator.wait_for_everyone()
    
    if args.with_tracking:        
        accelerator.end_training()


if __name__ == "__main__":
    main()
