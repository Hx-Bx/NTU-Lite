import torch


class KVCache:
    """Each layer's past keys and values, so a pass only runs the new tokens."""

    def __init__(self, config, max_tokens, device, dtype):
        n_kv = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        shape = (1, n_kv, max_tokens, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.max_tokens = max_tokens
        self.length = 0

    def update(self, i, k, v):
        start, end = self.length, self.length + k.shape[2]
        if end > self.max_tokens:
            raise RuntimeError(f"KV cache full ({self.max_tokens} tokens); the context cap comes in 4.4c")
        self.k[i][:, :, start:end] = k
        self.v[i][:, :, start:end] = v
        return self.k[i][:, :, :end], self.v[i][:, :, :end]

    def advance(self, n):
        self.length += n