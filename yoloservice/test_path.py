import torch

print(f"Pytorch版本：{torch.__version__}")
print(f"CUDA版本：{torch.version.cuda}")
print(f"cuDNN版本：{torch.backends.cudnn.version()}")

print(f"GPU设备数量：{torch.cuda.device_count()}")
print(f"GPU可用：{torch.cuda.is_available()}")

print(f"GPU设备：{torch.cuda.get_device_name(0)}")
print(f"当前显卡的总显存：{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3}GB")
