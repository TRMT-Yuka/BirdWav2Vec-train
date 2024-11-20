from accelerate import Accelerator
import torch

accelerator = Accelerator()
accelerator.print("Using device:", accelerator.device)

# トレーニングループ内で適宜確認
accelerator.print("Memory Allocated:", torch.cuda.memory_allocated())
accelerator.print("Memory Reserved:", torch.cuda.memory_reserved())


# import torch

print("Is CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.current_device() if torch.cuda.is_available() else "No CUDA")
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "No CUDA")