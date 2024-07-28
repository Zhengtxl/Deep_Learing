# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/17
# file:梯度消失和梯度爆炸.py
import  torch
import matplotlib.pyplot as  plot
from d2l import  torch as d2l
x=torch.arange(-8,8,0.1,requires_grad=True)
y=torch.sigmoid(x)
y.backward(torch.ones_like(x))
print(torch.ones_like(x))
d2l.plot(x.detach().numpy(),[y.detach().numpy(),x.grad.numpy()],legend=['sigmoid','gradient'])
plot.show()