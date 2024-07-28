# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/17
# file:dropout零实现和简洁版.py
import torch
from d2l import torch as d2l
from torch import nn


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
# 定义两层隐藏层，每层有256个单元
num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5


# 定义模型
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden1, num_hidden2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hidden1)
        self.lin2 = nn.Linear(num_hidden1, num_hidden2)
        self.lin3 = nn.Linear(num_hidden2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out
net = Net(num_inputs, num_outputs, num_hidden1, num_hidden2)
num_epochs, lr, batch_size = 10, 0.5, 256
# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')
# 定义数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 定义优化模型
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# 简洁版
# net1 = nn.Sequential(
#     nn.Flatten(),
#     nn.ReLU(),
#     # 在第一个全连接层之后添加一个dropout层
#     nn.Dropout(dropout1),
#     nn.Linear(256, 256),
#     nn.ReLU(),
#     # 在第二个全连接层之后添加一个dropout层
#     nn.Dropout(dropout2),
#     nn.Linear(256, 10)
# )
# def init_weight(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, std=0.01)
#
#
# net1.apply(init_weight)
# trainer1 = torch.optim.SGD(net1.parameters(), lr=lr)
# d2l.train_ch3(net1, train_iter, test_iter, loss, num_epochs, trainer1)
