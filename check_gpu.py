import torch

print("PyTorch", torch.__version__)
print("CUDA avaliable:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram:.1f} GB")