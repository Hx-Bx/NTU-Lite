import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(x.dtype)


def mlp(x, w):
    gate = F.linear(x, w["mlp.gate_proj.weight"])
    up = F.linear(x, w["mlp.up_proj.weight"])
    return F.linear(F.silu(gate) * up, w["mlp.down_proj.weight"])

def rope(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def attention(x, w, cos, sin, n_heads, n_kv_heads, cache=None, i=0):
    batch, seq, hidden = x.shape
    head_dim = hidden // n_heads

    q = F.linear(x, w["self_attn.q_proj.weight"], w["self_attn.q_proj.bias"])
    k = F.linear(x, w["self_attn.k_proj.weight"], w["self_attn.k_proj.bias"])
    v = F.linear(x, w["self_attn.v_proj.weight"], w["self_attn.v_proj.bias"])

    q = q.view(batch, seq, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq, n_kv_heads, head_dim).transpose(1, 2)

    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q = rope(q, cos, sin)
    k = rope(k, cos, sin)

    if cache is not None:
        k, v = cache.update(i, k, v)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=seq > 1, enable_gqa=True)
    out = out.transpose(1, 2).reshape(batch, seq, hidden)
    return F.linear(out, w["self_attn.o_proj.weight"])


def decoder_layer(x, w, cos, sin, config, cache=None, i=0):
    eps = config.rms_norm_eps
    h = rms_norm(x, w["input_layernorm.weight"], eps)
    x = x + attention(h, w, cos, sin, config.num_attention_heads, config.num_key_value_heads, cache, i)
    h = rms_norm(x, w["post_attention_layernorm.weight"], eps)
    return x + mlp(h, w)