import time
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding,
)

MODEL_DIR = "models/qwen2.5-0.5b"
device = "cuda"
dtype = torch.float16

config = AutoConfig.from_pretrained(MODEL_DIR, attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print("attention:", config._attn_implementation)

# Small parts that stay loaded the whole time
embed_w = load_file(f"{MODEL_DIR}/embed.safetensors", device=device)["weight"]

norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
norm.load_state_dict(load_file(f"{MODEL_DIR}/norm.safetensors"))
norm = norm.to(device, dtype)

rotary = Qwen2RotaryEmbedding(config=config).to(device)

# ONE empty layer "slot" on the GPU that we refill from disk
slot = Qwen2DecoderLayer(config, layer_idx=0).to(device, dtype)

def run_layer(i, hidden, positions, cos_sin):
    weights = load_file(f"{MODEL_DIR}/layer_{i:02d}.safetensors", device=device)
    slot.load_state_dict(weights, assign=True)
    out = slot(hidden, position_ids=positions, position_embeddings=cos_sin)
    return out[0] if isinstance(out, tuple) else out

@torch.no_grad()
def next_token(ids):
    hidden = F.embedding(ids, embed_w)
    positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    cos_sin = rotary(hidden, positions)

    for i in range(config.num_hidden_layers):
        hidden = run_layer(i, hidden, positions, cos_sin)

    hidden = norm(hidden)
    logits = F.linear(hidden[:, -1, :], embed_w)
    return logits.argmax(dim=-1)

stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
messages = [{"role": "user", "content": "Explain what a GPU is in one sentence."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

torch.cuda.reset_peak_memory_stats()
start = time.time()
count = 0

for _ in range(60):
    tok = next_token(ids)
    if tok.item() in stop_ids:
        break
    ids = torch.cat([ids, tok.unsqueeze(0)], dim=1)
    print(tokenizer.decode(tok), end="", flush=True)
    count += 1

elapsed = time.time() - start
print(f"\n\n{count} tokens in {elapsed:.1f}s ({count / elapsed:.1f} tokens/s)")
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")