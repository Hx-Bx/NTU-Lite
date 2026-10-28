# NTU-Lite

A local AI agent for Windows that runs a language model on the laptop you already use.
Model weights stay on the SSD and stream through the GPU one layer at a time, so it needs
very little RAM or VRAM.

## Status
Layer-streaming engine works: Qwen2.5-0.5B runs at 326 MB peak VRAM (vs 961 MB fully loaded)
with identical output. Next: speed optimizations (KV cache, prefetching, 4-bit).

## Setup (Windows, NVIDIA GPU)
```
py -V:3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r Req.txt
python scripts/split_model.py
```

## Run
```
python main.py "Explain what a GPU is in one sentence."
```

## Layout
- `Engine/` - streaming inference engine (`StreamedModel`, `generate`)
- `scripts/` - one-time tools (`split_model.py` cuts a model into per-layer files)
- `models/` - split model files (not committed)