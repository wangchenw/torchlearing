# -*- coding: utf-8 -*-
# @Time    : 2021/2/26 23:48
# @Author  : wangchen
# @FileName: day2.py
#%%matplotlib inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
#%%
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
#%%
print(features[0], labels[0])
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

# # 在../d2lzh_pytorch里面添加上面两个函数后就可以这样导入
# import sys
# sys.path.append("..")
# from d2lzh_pytorch import *

# set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);
plt.show()


#%%
plt.plot(features[:,1],labels.numpy(),12)
plt.show()

#%%
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 这一步将列表数据打乱
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break
#%%
import torch
##  实践出真知啊！！！
x = torch.rand(3,2,1)
print(x)
index = torch.LongTensor([0])
print(index)
# 如果想在x的第一个维度上选择x[2]和x[0]
y = torch.index_select(x, dim=1, index=index)
print(y)
# 如果想在x的第二个维度上选择，即x[...,2]和x[...,0]
#y = torch.index_select(x, dim=1, index=index)

# 另外，也可以用以下方法
#y = x.new()
#torch.index_select(x, dim=0, index=index, out=y)

#%%
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

#%%
# 将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
#  w和b参数求梯度来迭代参数的值
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
#%%线性回归的矢量计算表达式的实现。我们使用mm函数做矩阵乘法
def linreg(X, w, b):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    return torch.mm(X, w) + b
#%%定义损失函数
def squared_loss(y_hat, y):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2
#%% 定义优化算法 小批量随机梯度下降算法  它通过不断迭代模型参数来优化损失函数
def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data
#%%
import torch
x = torch.empty(5, 3)
print(x)
a = torch.empty([3, 4])
print("a = ", a)

b = torch.empty([2, 3])
print("b = ", b)
#%%
x=torch.empty([3,4])
print(x)

y=x[:,1]
x*=2
print(y)
print(x)