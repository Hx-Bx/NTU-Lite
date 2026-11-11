import sys
import psutil
import torch

from Engine.Generate import generate
from Engine.StreamedModel import StreamedModel

args = sys.argv[1:]
profile = "--profile" in args
device = "cpu" if "--cpu" in args else "cuda"
dtype = torch.float32 if "--fp32" in args else torch.float16
words = [a for a in args if not a.startswith("--")]
prompt = " ".join(words) or "Explain what a GPU is in one sentence."
model = StreamedModel("models/qwen2.5-0.5b", device=device, dtype=dtype, profile=profile)
print(f"device={device}  dtype={dtype}  threads={torch.get_num_threads()}\n")
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

text, stats = generate(model, prompt, on_token=lambda t: print(t, end="", flush=True))



speed = stats["tokens"] / stats["seconds"]
print(f"prefill (first pass): {stats['prefill']:.1f}s")
print(f"\n\n{stats['tokens']} tokens in {stats['seconds']:.1f}s ({speed:.1f} tokens/s)")
if device == "cuda":
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
print(f"peak RAM: {psutil.Process().memory_info().peak_wset / 1024**2:.0f} MB")

if profile:
    total = sum(model.timings.values())
    print("\ntime breakdown:")
    for name, secs in model.timings.items():
        print(f"  {name:8} {secs:6.1f}s  {secs / total * 100:5.1f}%")