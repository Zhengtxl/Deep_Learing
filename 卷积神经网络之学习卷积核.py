# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/27
# file:卷积神经网络之学习卷积核.py
import torch
from torch import nn
import  torch.nn.functional as F


def cord(X, K):
    # 计算两个矩阵实现互算,X是随机矩阵，K是权重
    h, d = K.shape
    # 假设每次步长为1，零补充为零
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - d + 1))
    # 循环遍历整个Y并与K相乘
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + d] * K).sum()
    return Y


# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = torch.ones((6, 8))
X[:, 2:6] = 0
# 真实参数为 1.0 -1.0
K = torch.tensor([[2.0, -2.0]])
Y = cord(X, K)
x = X.reshape((1, 1, 6, 8))
y = Y.reshape((1, 1, 6, 7))
etha = 3e-2  # 学习率
for i in range(50):
    # 获得随机的参数
    y_hat = conv2d(x)
    # 使用均方差计算loss损失函数
    loss = (y_hat - y) ** 2
    # 梯度归零
    conv2d.zero_grad()
    # 反向传播
    loss.sum().backward()
    # 更新权重
    conv2d.weight.data[:]-=etha*conv2d.weight.grad
    if (i+1)%2==0:
        print(f'epoch:{i+1},loss:{loss.sum():.4f},weight:{conv2d.weight.data}')
print(conv2d.weight.data.reshape(1,2))

