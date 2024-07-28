# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/19
# file:卷积神经网络之卷积层运算.py
# X表示原函数,W表示卷积层的核函数，权重函数
import torch


def cord(X, W):
    h, w = X.shape  # 矩阵X的长和宽
    h1, w1 = W.shape  # 矩阵W的长和宽

    # 默认步长为1，边缘填充为0，计算需要填充的长和宽
    h2, w2 = h - h1 + 1, w - w1 + 1
    # 定义长h2,宽w2，初始化全为零的矩阵
    y = torch.zeros((h2, w2))
    # 先行遍历，再列遍历
    for i in range(w2):
        for j in range(h2):
            y[i, j] = (X[i:i + h1, j:j + w1] * W).sum()
    return y


X = torch.tensor([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]])
K = torch.tensor([[1,0,1], [0,1,0],[1,0,1]])
result = cord(X, K)
print(result)
