import sys

import torch

from Engine.Generate import generate
from Engine.StreamedModel import StreamedModel

args = sys.argv[1:]
profile = "--profile" in args
words = [a for a in args if a != "--profile"]
prompt = " ".join(words) or "Explain what a GPU is in one sentence."
model = StreamedModel("models/qwen2.5-0.5b", profile=profile)
torch.cuda.reset_peak_memory_stats()

text, stats = generate(model, prompt, on_token=lambda t: print(t, end="", flush=True))

speed = stats["tokens"] / stats["seconds"]
print(f"\n\n{stats['tokens']} tokens in {stats['seconds']:.1f}s ({speed:.1f} tokens/s)")
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

if profile:
    total = sum(model.timings.values())
    print("\ntime breakdown:")
    for name, secs in model.timings.items():
        print(f"  {name:8} {secs:6.1f}s  {secs / total * 100:5.1f}%")