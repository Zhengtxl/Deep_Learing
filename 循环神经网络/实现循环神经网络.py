# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/31
# file:实现循环神经网络.py
import torch
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F

batch_size, num_steps = 32, 35
# 加载数据 加载《时间机器》的数据集
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
num_hiddens = 256
# 构建一个具有256个隐藏单元的单隐藏层的循环神经网络
rnn_layer = nn.RNN(len(vocab), num_hiddens)


class RNNModel(nn.Module):
    """循环神经网络模型"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        # 定义RNN层数，词汇表，隐藏单元
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 根据RNN层是否是双向的（这里不是），决定输出层的权重矩阵大小。
        # 如果不是双向RNN，则输出层权重矩阵的大小为(num_hiddens, vocab_size)；
        # 如果是双向RNN，则因为两个方向的隐藏状态会被拼接，所以大小为(num_hiddens * 2, vocab_size)
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        # 转换为one-hot编码
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态，直接返回全零张量作为隐藏状态的初始值
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
