# train.py
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter
from deepspeed.runtime.utils import move_to_device
from deepspeed.utils import groups
from torch import tensor
from transformers import AutoModelForCausalLM
import deepspeed
import deepspeed.comm as dist
import torch

model_name_or_path = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
max_length = 64
sequence_parallel_size = 2
micro_batch_size = 1

config_dict = {
    "train_micro_batch_size_per_gpu": 1,
    "zero_optimization": {
        "stage": 3,
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-3
        }
    },
    "sequence_parallel_size": sequence_parallel_size,
}

dtype = torch.bfloat16

# a simple Dataset
# replace with a real dataset but make sure `position_ids` are returned
input_ids = tensor([[1, 10, 10, 10, 2, 2], [1, 20, 20, 20, 2, 2]], )
position_ids = tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
ds = torch.utils.data.TensorDataset(input_ids, position_ids)
def collate_fn(batch):
    input_ids, position_ids = batch[0]
    return dict(input_ids=input_ids.unsqueeze(0),
                position_ids=position_ids.unsqueeze(0),
                labels=input_ids.unsqueeze(0))

dist.init_distributed(dist_backend='nccl', dist_init_required=True)

# Ulysses injection into HF Transformers
mpu = UlyssesSPAttentionHF.register_with_transformers(
    model_name_or_path=model_name_or_path,
    core_attn_implementation="sdpa",
    sequence_parallel_size=sequence_parallel_size,
    max_length=max_length,
    micro_batch_size=micro_batch_size,
    seq_length_is_variable=True,
)

# Deepspeed setup
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model, _, _, _ = deepspeed.initialize(config=config_dict,
                                        model=model,
                                        model_parameters=model.parameters(),
                                        mpu=mpu)

# UlyssesSPDataLoaderAdapter injection
sp_group = groups._get_sequence_parallel_group()
sp_world_size = groups._get_sequence_parallel_world_size()
sp_rank = groups._get_sequence_parallel_rank()
dl = torch.utils.data.DataLoader(ds, batch_size=micro_batch_size, collate_fn=collate_fn)
dl = UlyssesSPDataLoaderAdapter(
    dl,
    sp_rank=sp_rank,
    sp_group=sp_group,
    sp_world_size=sp_world_size,
    device=model.device,
)

# Normal training loop
for iter, batch in enumerate(dl):
    batch = move_to_device(batch, model.device)

    outputs = model(**batch)
    # as of this writing HF doesn't calculate loss with shift_labels yet and requires us to do it manually (liger does that automatically)
    shift_labels = batch["shift_labels"]
    loss = model.module.loss_function(
        logits=outputs.logits,
        labels=None,
        shift_labels=shift_labels,
        vocab_size=model.module.config.vocab_size,
    )

    # differentiable weighted per-shard-loss aggregation across ranks
    losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
    # special dealing with SFT that has prompt tokens that aren't used in loss computation
    good_tokens = sum((shift_labels != -100).view(-1))
    good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
    total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(sp_world_size))
    total_good_tokens = sum(good_tokens_per_rank)
    loss = total_loss / total_good_tokens

    if dist.get_rank() == 0:
        print(f"{iter}: {loss=}")

    model.backward(loss)