import torch
from time import time

a = torch.ones(1000)
b = torch.ones(1000)
start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)
start = time()
d = a + b
print(time() - start)

a =torch.ones(3)
b = 10
print(a + b)
print(a )
#%%

import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

#%%
num_inputs =2
num_examples =1000
true_w = [2,-3.14]
true_b = 4.2
features = torch.randn(num_examples,num_inputs,dtype=torch.float32)
print(features)
print(features.shape)
#%%
start = time()
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + b
labels += torch.tensor(np.random.normal(0,0.01, size=labels.size()), dtype=torch.float32)
print(time() - start)

#%%该方法不好 将矢量运算 转为标量运算
start = time()
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + b
labels1=[x+ 1  for x in labels]
print(time() - start)
#%%
print(features[0], labels[0])
#%%
def use_svg_display():
    #矢量图显示
    display.set_matplotlib_formats('svg')
def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    #设置图尺寸
    plt.rcParams['figure.figsize'] = figsize

# # 在../d2lzh_pytorch里面添加上面两个函数后就可以这样导入
# import sys
# sys.path.append("..")
# from d2lzh_pytorch import *

set_figsize()
plt.scatter(features[:,1],labels,1)

#%% 上面的函数不了解
print()
#%%
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

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);