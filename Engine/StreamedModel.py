from pathlib import Path

import torch
import time
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding,
)


class StreamedModel:
    """Runs a model one layer at a time, loading each layer from disk."""

    def __init__(self, model_dir, device="cuda", dtype=torch.float16, profile=False):
        self.dir = Path(model_dir)
        self.device = device
        self.dtype = dtype

        self.config = AutoConfig.from_pretrained(self.dir, attn_implementation="sdpa")
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir)

        # Small parts that stay loaded the whole time
        self.embed_w = load_file(str(self.dir / "embed.safetensors"), device=device)["weight"]

        self.norm = Qwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.norm.load_state_dict(load_file(str(self.dir / "norm.safetensors")))
        self.norm = self.norm.to(device, dtype)

        self.rotary = Qwen2RotaryEmbedding(config=self.config).to(device)

        # ONE layer-shaped slot on the GPU, refilled from disk for every layer
        self.slot = Qwen2DecoderLayer(self.config, layer_idx=0).to(device, dtype)

        self.num_layers = self.config.num_hidden_layers
        self.stop_ids = {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        }
        self.profile = profile
        self.timings = {"read": 0.0, "transfer": 0.0, "assign": 0.0, "compute": 0.0, "other": 0.0}

    def _sync(self):
        if self.profile:
            torch.cuda.synchronize()

    def _run_layer(self, i, hidden, positions, cos_sin):
        path = self.dir / f"layer_{i:02d}.safetensors"

        t0 = time.perf_counter()
        weights = load_file(str(path))                                   # file -> CPU RAM
        t1 = time.perf_counter()
        weights = {k: v.to(self.device) for k, v in weights.items()}    # CPU -> GPU
        self._sync()
        t2 = time.perf_counter()
        self.slot.load_state_dict(weights, assign=True)                  # into the slot
        self._sync()
        t3 = time.perf_counter()

        out = self.slot(hidden, position_ids=positions, position_embeddings=cos_sin)
        self._sync()
        t4 = time.perf_counter()

        self.timings["read"] += t1 - t0
        self.timings["transfer"] += t2 - t1
        self.timings["assign"] += t3 - t2
        self.timings["compute"] += t4 - t3
        return out[0] if isinstance(out, tuple) else out

    @torch.no_grad()
    def next_token(self, ids):
        t0 = time.perf_counter()
        hidden = F.embedding(ids, self.embed_w)
        positions = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
        cos_sin = self.rotary(hidden, positions)
        self._sync()
        self.timings["other"] += time.perf_counter() - t0

        for i in range(self.num_layers):
            hidden = self._run_layer(i, hidden, positions, cos_sin)

        t1 = time.perf_counter()
        hidden = self.norm(hidden)
        logits = F.linear(hidden[:, -1, :], self.embed_w)
        token = logits.argmax(dim=-1)
        self._sync()
        self.timings["other"] += time.perf_counter() - t1
        return token