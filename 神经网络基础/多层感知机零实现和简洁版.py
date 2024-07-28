# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/16
# file:多层感知机零实现和简洁版.py
import torch
from d2l import torch as d2l
from torch import nn

# 1、导入数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#  2、初始化模块,两层模型的权重和偏置初始化
num_inputs, num_outputs, num_hidden = 784, 10, 256
w1 = nn.Parameter(torch.randn(num_inputs, num_hidden, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))
w2 = nn.Parameter(torch.randn(num_hidden, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [w1, b1, w2, b2]


# 3、定义relu函数
def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)

# 4、定义模型
def net(x):
    x = x.reshape((-1, num_inputs))
    H = relu(x @ w1 + b1)
    return (H @ w2 + b2)


# 5、定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 6、进行模型训练
num_epochs, lr = 10, 0.1  # 确定迭代次数和学习率
updater = torch.optim.SGD(params=params, lr=lr)  # 参数和学习率的选定
d2l.train_ch3(net=net, train_iter=train_iter, test_iter=test_iter, loss=loss, num_epochs=num_epochs,
              updater=updater)  # 模型训练
d2l.predict_ch3(net,test_iter=test_iter)


# file:多层感知机简洁版.py
import torch
from d2l import torch as d2l
from torch import nn

# 添加了2个全连接层（之前我们只添加了1个全连接层。第一层是隐藏层，它包含256个隐藏单元，并使用了ReLU激活函数。第二层是输出层。
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std= 0.01)


net.apply(init_weight)
#
batch_size, lr, num_epochs = 256, 0.1, 10
# 获取数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 损失函数定义
loss = nn.CrossEntropyLoss(reduction='none')
# 优化策略，使用梯度下降
updater = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net=net, train_iter=train_iter, test_iter=test_iter, loss=loss, num_epochs=num_epochs, updater=updater)
