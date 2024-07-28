# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/16
# file:权重衰减零实现和简洁版.py
import torch
from d2l import torch as d2l

# 1、训练数据为20，测试数据为100，数据维度为200，小批量为5
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
# 随机生成权重和偏置 w和b
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
# 生成训练数据和训练数据迭代器
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size=batch_size)
# 生成测试数据和测试数据迭代器
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size=batch_size)


# 2、初始化w和b
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


#  3、定义 L2正则化
def L2_penalty(w):
    return torch.sum(w.pow(2)) / 2


# 4、代码训练
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * L2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())


train(lambd=3)

# file:权重衰减简洁版.py
import torch
from d2l import torch as d2l

# 1、训练数据为20，测试数据为100，数据维度为200，小批量为5
from torch import nn

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
# 随机生成权重和偏置 w和b
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
# 生成训练数据和训练数据迭代器
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size=batch_size)
# 生成测试数据和测试数据迭代器
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size=batch_size)


def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd}, {"params": net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))

    print('w的L2范数：', net[0].weight.norm().item())
train_concise(3)