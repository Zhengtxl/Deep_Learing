# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/30
# file:卷积神经网络之VGG.py
from d2l.torch import d2l
from torch import nn


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        # 核函数规格3*3，零填充为1，经过卷积层后，输出大小不变
        # 原来是112*112，输出后仍然是112*112，（112-3+1*2）/1+1=112
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        # 激活函数
        layers.append(nn.ReLU())
        in_channels = out_channels
        # 池化层采用最大汇聚的方式，汇聚层规格2*2，面试每次缩小为原来的四分之一
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# 这里的1是指重复多少次卷积层
conv_arch = ((1, 64), (1, 128), (1, 256), (1, 512), (1, 512))


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    # for循环遍历卷积层输入通道和输出通道
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
        # 返回模型
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))


ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size = 0.05, 10, 128
# 生成训练数据和测试数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
# 模型训练
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
