# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/29
# file:卷积神经网络之LeNet.py
import torch
from d2l import torch as d2l
from torch import nn

# 定义模型
net = nn.Sequential(
    # 卷积层
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    # 池化层
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 将数据铺平
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)
batch_size = 256
# 将mnist分为训练数据和测试数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    使用GPU计算模型在数据集上的精度
    net: 神经网络模型，预期是一个torch.nn.Module的实例。
    data_iter: 数据迭代器，用于迭代访问数据集中的样本。
    device: 指定计算应该在哪个设备上执行（CPU或GPU）。如果未指定，则自动从net的参数中推断出设备。
    """
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
    if not device:
        device = next(iter(net.parameters())).device  # 确定设备，确保数据和模型在同一个设备上
    # 用于累积两个值：正确预测的数量和总预测的数量。这个累积器在循环中用于计算准确率。
    metric = d2l.Accumulator(2)
    # 上下文管理器禁用梯度计算，在评估模式下不需要计算梯度。然后，遍历数据迭代器中的每一批数据
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            # 计算当前批次数据的准确率，并将其与当前批次中的样本数一起添加到累积器中。
            metric.add(d2l.accuracy(net(X), y), y.numel())
    # 返回准确率
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    # 初始化权重，使用Xavier均匀初始化方法，防止梯度爆炸或梯度消失
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    # 将模型移动到指定的设备中
    net.to(device)
    # 梯度下降优化方法
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 交叉熵损失函数
    loss = nn.CrossEntropyLoss()
    # 可视化训练过程中的损失和准确率
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 重置累计器，用于记录训练损失，训练精准率和样本数
        metric = d2l.Accumulator(3)
        # 将模式设置为训练模式
        net.train()
        """
        遍历训练数据迭代器：
        对每个批次的数据，首先将其移动到指定设备。
        前向传播，计算预测值。
        计算损失。
        反向传播，计算梯度。
        更新模型参数。
        在不计算梯度的情况下（torch.no_grad()），计算并累积损失、准确率和样本数。
        如果达到一定的批次间隔或到达最后一个批次，更新动画器以显示当前的训练损失和准确率。
        计算并显示测试集上的准确率。
        """
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))
        # 打印当前轮次的训练损失、训练准确率和测试准确率
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
        # 计算并打印平均每秒处理的样本数。
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')


lr, num_epochs = 0.9, 10
# d2l.try_gpu()检测是否含有GPU设备，没有的话,则改为CPU
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
