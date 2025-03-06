
def deepseek_v3_param():
    hidden_size = 2048
    num_layer = 19 # 27
    vocab_size = 163840
    intermediate_size = 11264
    moe_intermediate_sizes = 1024 #1408
    n_routed_experts = 32 # 64
    n_shared_experts = 2
    a = 16
    qk_nope_head_dim, qk_rope_head_dim = 128, 64
    kv_lora_rank = 512
    v_head_dim = 128
    
    embed_tokens = hidden_size * vocab_size
    lm_head = embed_tokens
    
    attn = (
        hidden_size * a * (qk_nope_head_dim+qk_rope_head_dim) + \
        hidden_size * (kv_lora_rank + qk_rope_head_dim) + \
        hidden_size + \
        kv_lora_rank * a * (qk_nope_head_dim+v_head_dim) + \
        a*v_head_dim * hidden_size
    )
    
    norm = hidden_size

    moe_layer = attn + (
        hidden_size * moe_intermediate_sizes*(n_routed_experts+n_shared_experts) * 3
    ) + norm * 2
    dense_layer = attn + (
        hidden_size * intermediate_size*3
    ) + norm * 2


    total = dense_layer + (num_layer-1)*(moe_layer) + embed_tokens+lm_head + norm
    print(f"moonshot/deepseek-16B model params: {total:_}, embedding: {embed_tokens+lm_head:_}, "
          f"dense params per layer: {dense_layer:_}, moe params per layer: {attn+moe_layer}")
    return total


def qwen2_5_3B_param():
    hidden_size = 2048
    a_q = 16
    a_kv = 2
    d = 2048 // 16
    intermediate_size = 11008
    num_layer = 36
    vocab_size = 151936

    embed_tokens = hidden_size * vocab_size
    lm_head = hidden_size * vocab_size

    attn = (
        hidden_size * a_q * d + \
        hidden_size * a_kv * d * 2 + \
        hidden_size * a_q * d
    )
    norm = hidden_size

    dense_layer = attn + (
        hidden_size * intermediate_size * 3
    ) + norm * 2

    total = num_layer * (dense_layer) + norm + embed_tokens + lm_head
    print(f"qwen2.5-3B model params: {total:_}, embedding: {embed_tokens+lm_head:_}, "
          f"dense params per layer: {dense_layer:_}")
    return total



def deepseek_v3_flops(batch_size, sequence_length):
    hidden_size = 2048
    num_layer = 27
    vocab_size = 163840
    intermediate_size = 11264
    moe_intermediate_sizes = 1408
    num_experts_per_tok = 6
    n_shared_experts = 2
    a = 16

    def attn_flops(batch_size, sequence_length):
        qk_nope_head_dim, qk_rope_head_dim = 128, 64
        kv_lora_rank = 512
        v_head_dim = 128

        q_proj = batch_size * sequence_length * hidden_size * (qk_nope_head_dim+qk_rope_head_dim)
        kv_a_proj_with_mqa = batch_size*sequence_length*hidden_size*(kv_lora_rank+qk_rope_head_dim)
        kv_b_proj = batch_size*sequence_length*kv_lora_rank*a*(qk_nope_head_dim+v_head_dim)
        attn = batch_size*sequence_length*sequence_length*a*(qk_nope_head_dim+qk_rope_head_dim) + batch_size*sequence_length*sequence_length*a*v_head_dim
        o_proj = batch_size*sequence_length*a*v_head_dim*hidden_size

        return 2 * (q_proj + kv_a_proj_with_mqa + kv_b_proj + attn + o_proj)


    def mlp_flops(batch_size, sequence_length, inter_size):
        return 6*batch_size*sequence_length*hidden_size*inter_size
    
    def dense_layer_flops(batch_size, sequence_length):
        return mlp_flops(batch_size, sequence_length, intermediate_size)

    def moe_layer_flops(batch_size, sequence_length):
        routed = mlp_flops(batch_size, sequence_length, moe_intermediate_sizes*num_experts_per_tok)
        shared = mlp_flops(batch_size, sequence_length, moe_intermediate_sizes*n_shared_experts)
        return routed+shared

    total = (
        attn_flops(batch_size, sequence_length) + dense_layer_flops(batch_size, sequence_length) \
        + (num_layer-1) * (attn_flops(batch_size, sequence_length)+moe_layer_flops(batch_size, sequence_length)) \
        + 2* batch_size * sequence_length * vocab_size * hidden_size
    )
    print(f"moonshot/deepseek-16B model flops: {total:_}")
    return total


def qwen2_5_3B_flops(batch_size, sequence_length):
    hidden_size = 2048
    a_q = 16
    a_kv = 2
    d = 2048 // 16
    intermediate_size = 11008
    num_layer = 36
    vocab_size = 151936

    def attn_flops(batch_size, sequence_length):
        qkv = 2*batch_size*sequence_length*hidden_size*(a_q*d + 2*a_kv*d) 
        attn = 4*batch_size*hidden_size*sequence_length*sequence_length
        o = 2*batch_size*sequence_length*hidden_size*hidden_size
        return qkv+attn+o

    def mlp_flops(batch_size, sequence_length):
        return 6*batch_size*sequence_length*hidden_size*intermediate_size
    
    total = num_layer * (attn_flops(batch_size, sequence_length)+mlp_flops(batch_size, sequence_length)) + 2* batch_size * sequence_length * vocab_size * hidden_size
    print(f"qwen2.5-3B model flops: {total:_}")
    return total


def calculate_flops(model_name_or_path, batch_size, sequence_length):
    if "qwen2.5-3B" in model_name_or_path:
        return qwen2_5_3B_flops(batch_size, sequence_length)
    else:
        return deepseek_v3_flops(batch_size, sequence_length)


if __name__ == "__main__":
    calculate_flops("qwen2.5-3B", 1, 4096)
    calculate_flops("moonshot/deepseek", 1, 4096)
    print(f'{calculate_flops("qwen2.5-3B", 1, 4096)/calculate_flops("moonshot/deepseek", 1, 4096)}')
    
    deepseek_v3_param()
    qwen2_5_3B_param()
