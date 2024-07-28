# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/17
# file:MLP的基本概念.py
import torch
from torch import nn
from torch.nn import functional as F
# 使用内置的函数块
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
x = torch.randn(2, 20)
print('使用内置的函数块',net(x))
# 自定义块
class MLP(nn.Module):
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, x):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(x)))
net1=MLP()
print('使用自定义的函数块',net1(x))