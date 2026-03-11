import torch
print(torch.cuda.is_available())          # 看有没有GPU
print(torch.cuda.device_count())          # 有几个GPU
print(torch.cuda.get_device_name(0))      # GPU的名字
print(torch.cuda.get_device_properties(0)) # GPU详细信息
print(1)
import torch
print(torch.__version__)          # PyTorch版本
print(torch.version.cuda)         # 这个PyTorch包是为了哪个CUDA版本编译的？

import torch
import transformers
import torchvision
print("✅ 所有库都安装成功!")
