from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding,
)


class StreamedModel:
    """Runs a model one layer at a time, loading each layer from disk."""

    def __init__(self, model_dir, device="cuda", dtype=torch.float16):
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

    def _run_layer(self, i, hidden, positions, cos_sin):
        path = self.dir / f"layer_{i:02d}.safetensors"
        weights = load_file(str(path), device=self.device)
        self.slot.load_state_dict(weights, assign=True)
        out = self.slot(hidden, position_ids=positions, position_embeddings=cos_sin)
        return out[0] if isinstance(out, tuple) else out

    @torch.no_grad()
    def next_token(self, ids):
        hidden = F.embedding(ids, self.embed_w)
        positions = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
        cos_sin = self.rotary(hidden, positions)

        for i in range(self.num_layers):
            hidden = self._run_layer(i, hidden, positions, cos_sin)

        hidden = self.norm(hidden)
        logits = F.linear(hidden[:, -1, :], self.embed_w)
        return logits.argmax(dim=-1)