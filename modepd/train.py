"""
Reference: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
"""
import argparse
import logging
import os
from tqdm.auto import tqdm
import math

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import datasets
import transformers
from transformers import (
    DataCollatorForLanguageModeling,
    get_scheduler,
)

from modepd.utils import register_custom_model, prepare_model_and_tokenizer, get_memory_stats, build_dataset


logger = get_logger(__name__)
register_custom_model()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_to", type=str, default="tensorboard",)
    parser.add_argument("--output_dir", type=str, default=None,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",)

    parser.add_argument("--dataset_name_or_path", type=str, default="HuggingFaceFW/fineweb",) # "allenai/OLMoE-mix-0924"
    parser.add_argument("--dataset_config_name", type=str, default=None,) # None
    parser.add_argument("--data_type", type=str, default=None)
    parser.add_argument("--streaming_dataset", action='store_true')
    parser.add_argument("--validation_dataset_name_or_path", type=str, default="Salesforce/wikitext")
    parser.add_argument("--validation_dataset_config_name", type=str, default="wikitext-2-raw-v1",) # None
    parser.add_argument("--evaluate_every", type=int, default=100)

    parser.add_argument("--block_size", type=int, default=4*1024,)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,)
    parser.add_argument("--skip_train", action='store_true')

    parser.add_argument("--weight_decay", type=float, default=0.1,)
    parser.add_argument("--learning_rate", type=float, default=4e-4,)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",)
    parser.add_argument("--num_warmup_steps", type=int, default=0,)
    parser.add_argument("--max_train_steps", type=int, default=5,)
    parser.add_argument("--checkpointing_steps", type=int, default=-1,)
    parser.add_argument("--with_tracking", action="store_true",)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,)
    parser.add_argument("--push_to_hub", action="store_true",)

    parser.add_argument("--mod_type", type=str, default=None, choices=['staged', 'integrated'])
    parser.add_argument("--staged_mod_topk", type=int, default=2048)
    parser.add_argument("--finetune_mod_only", action="store_true",)

    return parser.parse_args()


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
        zero_stage=2,
        zero3_save_16bit_model=True,
    )
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

    alloc, max_alloc, reserved, max_reserved = get_memory_stats()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: {trainable_params/1024**3:.2f} B, device: {model.device}, dtype: {model.dtype}"
        f", Memory stats after initializing model: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {reserved:.2f} G / {max_reserved:.2f} G", 
        main_process_only=False)
    logger.info(
        f"Model emb size: {model.get_input_embeddings().weight.shape[0]}"
        f", tokenizer vocab size: {len(tokenizer)}")
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # assert False, f"Tokenizer vocab size {len(tokenizer)} is larger than the model embedding size {embedding_size}."
    
    # 3. DataLoaders creation
    train_dataset = build_dataset(
        args.dataset_name_or_path, args.dataset_config_name, args.streaming_dataset, tokenizer, 'train', 
        args.data_type, args.block_size, logger, accelerator)
    validation_dataset = build_dataset(
        args.validation_dataset_name_or_path, args.validation_dataset_config_name, args.streaming_dataset, tokenizer, 'validation', 
        block_size=args.block_size, logger=logger, accelerator=accelerator)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(
        train_dataset,
        # shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        batch_size=args.per_device_train_batch_size,
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
    num_update_steps_per_epoch = args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
    )
    alloc, max_alloc, reserved, max_reserved = get_memory_stats()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{model}")
    logger.info(
        f"Model parameters: {trainable_params/1024**3:.2f} B, device: {model.device}, dtype: {model.dtype}"
        f", Memory stats before training: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {reserved:.2f} G / {max_reserved:.2f} G"
        , main_process_only=False)
    for idx, batch in enumerate(train_dataloader):
        logger.info(f"rank {accelerator.process_index} batch {idx}: {batch['input_ids'][0, :5].tolist()}", main_process_only=False)
        if idx == 2:
            break

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        accelerator.init_trackers("moe-pruning", experiment_config)

    completed_steps = 0
    if not args.skip_train:
        #################
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        # logger.info(f"  Num examples = {len(train_dataset)}")
        # logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        resume_step = None
        # starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
            dirs = [f.path for f in os.scandir(args.resume_from_checkpoint) if f.is_dir() and f.name.startswith("step_")]
            if len(dirs) > 0:
                dirs.sort(key=os.path.getctime)
                checkpoint_path = dirs[-1]
                path = os.path.basename(checkpoint_path)

                accelerator.load_state(checkpoint_path)
                # Extract `step_{i}`
                training_difference = os.path.splitext(path)[0]

                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
                completed_steps = resume_step // args.gradient_accumulation_steps
                logger.info(
                    f"Resumed from checkpoint: {checkpoint_path}, resume steps (w. grad acc): {resume_step}, "
                    f"completed_steps: {completed_steps}"
                )
            else:
                logger.warning(
                    f"Please be aware that resume_from_checkpoint is specified as {args.resume_from_checkpoint}, "
                    f"but no ckpt is detected"
                )

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

        model.train()
        if args.with_tracking:
            step_loss = torch.zeros(1, device=model.device, dtype=torch.float)
        if args.resume_from_checkpoint and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                if args.with_tracking:
                    step_loss += loss.detach().float()
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
                    # logger.info(f"completed_steps {completed_steps}: loss: {global_loss}, lr: {lr_scheduler.get_last_lr()}")
                    log_info = {"train_loss": global_loss.item(),}
                    for lr_idx, lr in enumerate(lr_scheduler.get_last_lr()):
                        log_info[f"lr_{lr_idx}"] = lr
                    
                    accelerator.log(log_info, step=completed_steps)
                    step_loss.zero_()

                if args.checkpointing_steps > 0 and completed_steps % args.checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                if completed_steps % args.evaluate_every == 0:
                    losses = []
                    for step, batch in tqdm(
                        enumerate(validation_dataloader), desc=f"Evaluation at step {completed_steps}",
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
                            
            if completed_steps >= args.max_train_steps:
                break
    
    losses = []
    for step, batch in tqdm(
        enumerate(validation_dataloader), desc=f"Evaluation at step {completed_steps}",
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
            # with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            #     json.dump({"perplexity": perplexity}, f)
        logger.info(f"Saving model to {args.output_dir} done!", main_process_only=False)
        accelerator.wait_for_everyone()
    
    if args.with_tracking:        
        accelerator.end_training()


if __name__ == "__main__":
    main()
