import sys

import torch

from Engine.Generate import generate
from Engine.StreamedModel import StreamedModel

prompt = " ".join(sys.argv[1:]) or "Explain what a GPU is in one sentence"

model = StreamedModel("models/qwen2.5-0.5b")
torch.cuda.reset_peak_memory_stats()

text, stats = generate(model, prompt, on_token=lambda t: print(t, end="", flush=True))

speed = stats["tokens"] / stats["seconds"]
print(f"\n\n{stats['tokens']} tokens in {stats['seconds']:.1f}s ({speed:.1f} tokens/s)")
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")