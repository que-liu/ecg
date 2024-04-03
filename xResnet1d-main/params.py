# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:27:12 2023

@author: hawkiyc
"""

#%%
'Import Libraries'

import os
import torch
import torch.nn as nn

#%%
'Hyperparameters'

activation = nn.LeakyReLU(inplace=True)
w_size = 2.5 # Random Segmentation Size (Unit: seconds)
model_sr = 100 #Sampling Rate for Model Input     #模型输入的采样率
stem_k, block_k = 7, 5 # Kernel Size for Stem and ResBlock Conv1d   茎秆和 ResBlock Conv1d 的内核尺寸
data_dim = 12 # Dimension for Model Input      模型输入尺寸
batch_size =64
model_dropout = None # xResNet Dropout Rate
fc_drop = None # FC Layer Dropout Rate
out_activation = nn.Sigmoid()
loss_fn = torch.nn.BCEWithLogitsLoss()
n_epochs = 100
#%%
"Setting GPU"

use_cpu = False
m_seed = None # Set Seed for Reproducibility   为可重复性播下种子

if use_cpu:
    device = torch.device('cpu')

elif torch.cuda.is_available(): 
    device = torch.device('cuda')
    if m_seed:
        torch.cuda.manual_seed(m_seed)
    torch.cuda.empty_cache()

elif not torch.backends.mps.is_available(): #Setting GPU for Mac User
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was "
              "NOT built with MPS enabled.")
    else:
        print("MPS not available because this MacOS version is NOT 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = torch.device("mps")
print(device)

#%%
'Make Output Dir'
"""
这段代码用于配置PyTorch操作的设备（CPU、GPU或MPS）。这里对use_cpu、CUDA可用性以及MPS可用性进行了检查，然后根据情况设置设备。
如果use_cpu为True，则将device设置为torch.device('cpu')，表示使用CPU进行计算。
如果CUDA可用（即torch.cuda.is_available()为True），则将device设置为torch.device('cuda')，表示使用GPU进行计算。
如果设置了m_seed，则调用torch.cuda.manual_seed(m_seed)设置GPU随机种子，并调用torch.cuda.empty_cache()释放GPU内存缓存。
如果CUDA不可用，但torch.backends.mps.is_available()为False，则尝试使用MPS（Multi-Process Service）。首先检查是否构建了MPS支持，如果没有构建，则打印一条消息说明PyTorch未启用MPS。如果已构建但MPS在该macOS版本上不可用或该机器上没有启用MPS的设备，则打印相应的消息。
4.最后，如果以上条件都不满足，则将device设置为torch.device("mps")，表示使用MPS进行计算。
总之，这段代码根据不同条件选择合适的设备（CPU、GPU或MPS）来执行PyTorch操作。"""
if not os.path.isdir('results'):
    os.makedirs('results')
