"""
最简单的使用示例
"""

import torch
import torch.nn as nn
from sigreg_loss import SIGReg


# 定义模型
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128)
)

# 创建 SIGReg 损失
sigreg = SIGReg()

# 创建数据
x = torch.randn(64, 784)  # batch_size=64, input_dim=784

# 前向传播
embeddings = model(x)  # [64, 128]

# 计算损失
loss = sigreg(embeddings)
print(f"Loss: {loss.item():.6f}")

# 反向传播
loss.backward()
print("Gradients computed!")
