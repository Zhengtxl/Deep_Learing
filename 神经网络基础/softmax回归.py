import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

batch_size = 256  # 批量大小

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 读取数据集
"""
torch.nn.Flatten(start_dim=1, end_dim=- 1)用于设置网络中的展平层，常用于将输入张量展平为二维张量，也可以指定展平维度的范围[start_dim, end_dim]。
torch.nn.Linear(in_features, out_features)用于设置网络中的全连接层，需要注意的是全连接层的输入与输出都是二维张量。
in_features指的是输入的二维张量的第二个维度的大小。
out_features指的是输出的二维张量的第二个维度的大小。
torch.nn.Sequential()是PyTorch中的一个类，它允许用户将多个计算层按照顺序组合成一个模型。
"""
from torch import nn

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


# 初始化权重参数
def init_weights(m):  # m就是当前的层layer
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)  # w从均值为0（默认），标准差为0.01的正态分布随机选取
        m.bias.data.fill_(0)  # b设为0


net.apply(init_weights)  # 把init_weights函数apply到net上面

# 损失函数定义
loss = nn.CrossEntropyLoss()
# 定义优化算法  使用学习率为0.1的小批量随机梯度下降作为优化算法。
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# 训练数据
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


# 预测
def predict_ch3(net, test_iter, n=6):  # @save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:  # 拿出一个批量的样本
        break
    trues = d2l.get_fashion_mnist_labels(y)  # 实际标签
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))  # 预测，取最大概率的类别的标签
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show()


predict_ch3(net, test_iter)
