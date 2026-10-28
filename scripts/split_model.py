import os
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = "models/qwen2.5-0.5b"

os.makedirs(OUT_DIR, exist_ok=True)

# Load on the CPU (no .to("cuda")) - we're only cutting it up
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def save(tensors, name):
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    save_file(tensors, os.path.join(OUT_DIR, name))
    mb = sum(t.numel() * t.element_size() for t in tensors.values()) / 1024**2
    print(f"saved {name:22} {mb:7.1f} MB")

save({"weight": model.model.embed_tokens.weight}, "embed.safetensors")
for i, layer in enumerate(model.model.layers):
    save(layer.state_dict(), f"layer_{i:02d}.safetensors")
save({"weight": model.model.norm.weight}, "norm.safetensors")

tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
print("lm_head reuses embed weights:", tied)

model.config.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("done")