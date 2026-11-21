"""python check_gpu.py  -> tells whether PyTorch can use your NVIDIA GPU (used by step 11 only)"""
import time, torch
print("torch", torch.__version__, "| built with CUDA:", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    a = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize(); t = time.time()
    for _ in range(10): a @ a
    torch.cuda.synchronize(); print(f"10 matmuls 4096x4096 on GPU: {time.time() - t:.2f} s -> GPU works")
else:
    print("No GPU used. If you have an NVIDIA GPU you most likely have the CPU-only PyTorch build:")
    print("  1) run  nvidia-smi  (top right shows the max CUDA version your driver supports)")
    print("  2) open https://pytorch.org/get-started/locally/  -> Windows, Pip, pick the CUDA version <= that")
    print("  3) pip uninstall torch -y   and then run the pip command shown there (it contains --index-url .../whl/cuXXX)")
