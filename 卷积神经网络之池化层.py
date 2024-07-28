# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/28
# file:卷积神经网络之池化层.py
import torch


def pool2d(X, pool_size, mode='max'):
    # 确保输入是浮点数
    X = X.float()
    pool_h, pool_w = pool_size
    Y = torch.zeros(X.shape[0] - pool_h + 1, X.shape[1] - pool_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + pool_h, j:j + pool_w].max()
            if mode == 'avg':
                Y[i, j] = X[i:i + pool_h, j:j + pool_w].mean()
    return Y


X = torch.arange(0, 9).reshape(3, 3)
print(X)
result=pool2d(X,(2,2))
print(result)
result=pool2d(X,(2,2),'avg')
print(result)