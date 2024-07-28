# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/27
# file:卷积神经网络之填充和步幅.py
import torch
import torch.nn.functional as F


def cord(X, K, padding=1,stride=1):
    h, d = K.shape
    x_h, x_d = X.shape
    x_h1 = int((x_h + padding * 2 - h )/stride+1)
    x_d1 = int((x_d + padding * 2 - d )/stride+1)
    X_padded = F.pad(X, (padding, padding, padding, padding), mode='constant', value=0)
    print(X_padded)
    Y = torch.zeros((x_h1, x_d1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X_padded[i*stride:i*stride + h, j*stride:j*stride + d] * K).sum()
    return Y


X = torch.arange(0, 9).reshape(3, 3)
K = torch.arange(0, 4).reshape(2, 2)

# 简便操作
import torch
from torch import nn
# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])
# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1,stride=1)
X = torch.rand(size=(8, 8))
print(X.shape)
Y=comp_conv2d(conv2d, X)
print(Y.shape)