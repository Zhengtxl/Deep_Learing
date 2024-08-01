# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/30
# file:卷积神经网络之ResNet.py
from d2l.torch import d2l
from torch import nn
from torch.nn import functional as F

"""
残差块里首先有2个有相同输出通道数的3 × 3卷积层。每个卷积
层后接一个批量规范化层和ReLU激活函数。然后我们通过跨层数据通路，跳过这2个卷积运算，将输入直接
加在最后的ReLU激活函数前。这样的设计要求2个卷积层的输出与输入形状一样，从而使它们可以相加。如
果想改变通道数，就需要引入一个额外的1 × 1卷积层来将输入变换成需要的形状后再做相加运算
"""


class Residual(nn.Module):
    # use_1x1conv：一个布尔值，指示是否使用1x1卷积来调整输入X的维度，以便在将X加到Y上时，它们的维度能够匹配。
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        # 两个3x3卷积层，用来提取特征
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        # 仅当use_1x1conv=True时存在，它的作用是调整输入X的通道数或空间维度，以便在加法操作中与Y的维度相匹配。
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        # bn1,bn2应用于批量规范化，以加速训练过程和改善泛化能力
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    # 前向传播
    def forward(self, X):
        # 输入X通过self.conv1，然后是ReLU激活函数和self.bn1进行批量归一化
        Y = F.relu(self.bn1(self.conv1(X)))
        # 将输出通过conv2和bn2
        Y = self.bn2(self.conv2(Y))
        # 如果存在conv3则进行残差连接
        if self.conv3:
            X = self.conv3(X)
        Y += X
        # 应用到relu激活函数中
        return F.relu(Y)


# 存储需要构建的残差块
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    """
    对于第一个残差块（且不是网络的第一个残差块），它使用 Residual 类创建一个新的残差块实例，
    输入通道数为 input_channels，输出通道数为 num_channels，并设置 use_1x1conv=True 和 strides=2。
    这通常用于在残差网络的早期阶段减少特征图的尺寸（高度和宽度减半），同时增加通道数。
    """
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


# 在输出通道数为64、步幅为2的7 × 7卷积层后，接步幅为2的3 × 3的最大汇聚层。ResNet每个卷积层后增加了批量规范化层。
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# b2、b3、b4、b5分别是不同深度的残差块序列，它们通过resnet_block函数生成，并使用nn.Sequential进行封装
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
net = nn.Sequential(
    b1,  # 初始的卷积层、批量归一化、ReLU激活和最大池化层
    b2,  # 第一个残差块序列，输出通道数为64
    b3,  # 第二个残差块序列，输出通道数增加到128
    b4,  # 第三个残差块序列，输出通道数增加到256
    b5,  # 第四个残差块序列，输出通道数增加到512
    nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化层，将特征图的大小调整为1x1
    nn.Flatten(),  # 扁平化层，将多维的输入一维化，准备输入到全连接层
    nn.Linear(512, 10)  # 全连接层，将512维的特征转换为10维的输出，对应10个类别的得分
)
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
