from pathlib import Path

import torch
import time
import json
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from Engine.Layer import decoder_layer, rms_norm
from Engine.KVCache import KVCache


class StreamedModel:
    """Runs a model one layer at a time, loading each layer from disk."""

    def __init__(self, model_dir, device="cuda", dtype=torch.float16, profile=False):
        self.dir = Path(model_dir)
        self.device = device
        self.dtype = dtype

        self.config = AutoConfig.from_pretrained(self.dir, attn_implementation="sdpa")
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir)

        # Small parts that stay loaded the whole time
        self.embed_w = load_file(str(self.dir / "embed.safetensors"), device=device)["weight"].to(device,dtype)

        self.norm_w = load_file(str(self.dir / "norm.safetensors"))["weight"].to(device, dtype)

        self.rotary = Qwen2RotaryEmbedding(config=self.config).to(device)

        self.num_layers = self.config.num_hidden_layers
        self.stop_ids = {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        }
        self.profile = profile
        self.timings = {"read": 0.0, "transfer": 0.0, "compute": 0.0, "other": 0.0}

    def _sync(self):
        if self.profile and self.device == "cuda":
            torch.cuda.synchronize()

    def new_cache(self, max_tokens=2048):
        return KVCache(self.config, max_tokens, self.device, self.dtype)

    def _run_layer(self, i, hidden, cos, sin, cache):
        path = self.dir / f"layer_{i:02d}.safetensors"

        t0 = time.perf_counter()
        weights = load_file(str(path))                                             # file -> RAM (mapped)
        t1 = time.perf_counter()
        weights = {k: v.to(self.device, self.dtype) for k, v in weights.items()}  # to device / dtype
        self._sync()
        t2 = time.perf_counter()

        hidden = decoder_layer(hidden, weights, cos, sin, self.config, cache, i)
        self._sync()
        t3 = time.perf_counter()

        self.timings["read"] += t1 - t0
        self.timings["transfer"] += t2 - t1
        self.timings["compute"] += t3 - t2
        return hidden

    
    @torch.no_grad()
    def next_token(self, ids, cache):
        t0 = time.perf_counter()
        hidden = F.embedding(ids, self.embed_w)
        start = cache.length
        positions = torch.arange(start, start + ids.shape[1], device=self.device).unsqueeze(0)
        cos, sin = self.rotary(hidden, positions)
        self._sync()
        self.timings["other"] += time.perf_counter() - t0

        for i in range(self.num_layers):
            hidden = self._run_layer(i, hidden, cos, sin, cache)
        cache.advance(ids.shape[1])

        t1 = time.perf_counter()
        hidden = rms_norm(hidden, self.norm_w, self.config.rms_norm_eps)
        logits = F.linear(hidden[:, -1, :], self.embed_w)
        token = logits.argmax(dim=-1)
        self._sync()
        self.timings["other"] += time.perf_counter() - t1
        return token

