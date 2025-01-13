"""
Reference: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
"""
import argparse
from itertools import chain
import logging
import os
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import datasets
from datasets import (
    load_dataset, 
    # interleave_datasets,
)
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig, 
    # DataCollatorForLanguageModeling,
    default_data_collator,
    get_scheduler,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_to", type=str, default="tensorboard",)
    parser.add_argument("--output_dir", type=str, default=None,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None,)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb",)
    parser.add_argument("--dataset_config_name", type=str, default="sample-350BT",)
    parser.add_argument("--block_size", type=int, default=16*1024,)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,)

    parser.add_argument("--weight_decay", type=float, default=0.01,)
    parser.add_argument("--learning_rate", type=float, default=5e-5,)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",)
    parser.add_argument("--num_warmup_steps", type=int, default=0,)
    parser.add_argument("--max_train_steps", type=int, default=5,)
    parser.add_argument("--checkpointing_steps", type=int, default=None,)
    parser.add_argument("--with_tracking", action="store_true",)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,)
    parser.add_argument("--push_to_hub", action="store_true",)

    # print(f"from os environ: rank: {os.environ.get('RANK', None)}/{os.environ.get('WORLD_SIZE', None)}"
    #       f", local rank: {os.environ.get('LOCAL_RANK', None)}/{os.environ.get('LOCAL_WORLD_SIZE', None)}"
    #       f", node rank: {os.environ.get('NODE_RANK', None)}"
    #       f", master addr: {os.environ.get('MASTER_ADDR', None)}, port: {os.environ.get('MASTER_PORT', None)}"
    # )
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

    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy="FULL_SHARD",
        activation_checkpointing=True,
        auto_wrap_policy="transformer_based_wrap",
        mixed_precision_policy=torch.distributed.fsdp.MixedPrecision(param_dtype=torch.bfloat16),
        # cpu_offload=True,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_plugin=fsdp_plugin,
        **accelerator_log_kwargs
    )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
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
    # model_name = "deepseek-ai/DeepSeek-V2-Lite"
    model_name = 'Qwen/Qwen2.5-3B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, use_cache=False).cuda()
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    print(f"model emb size: {model.get_input_embeddings().weight.shape[0]}"
          f", tokenizer vocab size: {len(tokenizer)}")
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        assert False, f"Tokenizer vocab size {len(tokenizer)} is larger than the model embedding size {embedding_size}."
    
    #################
    # Prepare dataset
    # TODO: mix fineweb (en) and fineweb-2 (zh) dataset
    # en = load_dataset("HuggingFaceFW/fineweb", name="sample-350BT", split="train", streaming=True)
    # zh = load_dataset("HuggingFaceFW/fineweb-2", name="cmn_Hani", split="train", streaming=True)
    # ds = interleave_datasets([en, zh], probabilities=[0.8, 0.2], seed=42)
    # ds = en
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, streaming=True)
    
    #################
    # Preprocessing the datasets.
    # 1. Only load text fields for the dataloader
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

    # 2. Padding to max length    
    if args.block_size is None:
        block_size = tokenizer.model_max_length
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )
    
    train_dataset = lm_datasets["train"]
    # fine-web doesn't have validation set
    # eval_dataset = lm_datasets["validation"]
    # this logic will be handled by `accelerator.prepare`
    # if accelerator.num_processes > 1:
    #     train_dataset = split_dataset_by_node(train_dataset, rank=accelerator.process_index, world_size=accelerator.num_processes)

    # Log a few random samples from the training set:
    data_iter = iter(train_dataset)
    for index in range(3):
        sample = next(data_iter)
        logger.info(f"rank: {accelerator.process_index}/{accelerator.num_processes} sample {index}: {sample['input_ids'][:5]}", main_process_only=False)
    
    # 3. DataLoaders creation
    train_dataloader = DataLoader(
        train_dataset,
        # shuffle=True,
        # collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
    )

    for batch in train_dataloader:
        print(batch['input_ids'].shape)
        break

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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    logger.info(f"Model dtype: {model.dtype}, device: {model.device}")
    for idx, batch in enumerate(train_dataloader):
        print(f"rank {accelerator.process_index} batch {idx}: {batch['input_ids'][0, :5].tolist()}")
        if idx == 2:
            break
    
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

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
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        # need to multiply `gradient_accumulation_steps` to reflect real steps
        resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
        starting_epoch = resume_step // len(train_dataloader)
        completed_steps = resume_step // args.gradient_accumulation_steps
        resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    model.train()
    if args.with_tracking:
        total_loss = 0.
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
                total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1
        
        if isinstance(checkpointing_steps, int) and checkpointing_steps > 0:
            if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                output_dir = f"step_{completed_steps}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)
        
        if args.with_tracking:
            accelerator.log(
                {
                # "perplexity": perplexity
                "train_loss": total_loss.item() / completed_steps,
                "step": completed_steps,
                },
                step=completed_steps,
            )
        
        if completed_steps >= args.max_train_steps:
            break
    
    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
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


if __name__ == "__main__":
    main()
