# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/28
# file:卷积神经网络之多输入和多通道.py
# 多输入
import torch
from d2l import  torch as d2l

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
def mult(X,K):
    return sum(d2l.corr2d(x,k) for x,k in zip(X,K))
print(mult(X,K))

# 多通道
def corr2d_mult2(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([mult(X, k) for k in K], 0)
# 将原来的一个通道转换为三通道
K = torch.stack((K, K + 1, K + 2), 0)
print(K)
print(K.shape)
print(corr2d_mult2(X,K))