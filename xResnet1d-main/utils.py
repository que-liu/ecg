# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:09:04 2023

@author: Revlis_user
"""
#%%
'Import Libraries'

import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random
from sklearn.metrics import precision_recall_curve
import torch
from torch.utils.data import Dataset

from params import *
from preprocessing import *

#%%
'Data Loader'




class Dataset(Dataset):
    
    # Random Crop Signal to Segment with w_size seconds length#     以 w_size 秒为长度的随机裁剪信号到分段
    def __init__(self, data, labels, window_size,):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the full record and its corresponding label # 获取完整记录及其相应标签
        record = self.data[index]
        label = self.labels[index]

        # Randomly select a segment of fixed length from the full record# 从完整记录中随机选择一个固定长度的片段
        start = random.randint(0, record.shape[-1] - self.window_size)
        segment = record[:,start : start + self.window_size]

        return segment, label


"""
定义StepWiseTestset类，用于处理元素最大值测试集。
__init__(self, data, labels, window_size, stride)：类的初始化方法，
接收数据集data、标签集labels、窗口大小window_size和步长stride作为参数，并将它们保存在类的属性中。
__getitem__(self, index)：
根据给定的索引index获取数据集中的一个样本。
首先，从数据集中获取完整的记录record和对应的标签label。
然后，根据窗口大小和步长，创建具有重叠窗口的分段segments，将这些分段作为样本的特征，同时将标签作为样本的标签。
最后，返回这些分段和标签作为样本的内容。
这段代码的作用是将输入的数据集切分成具有重叠窗口的分段，用于元素最大值测试集。每个分段是一个窗口大小的子集，通过以步长stride滑动窗口来创建这些分段。"""
"""
1.__init__(self, data, labels, window_size, stride)：初始化方法接收四个参数：data表示输入数据，labels表示对应的标签，window_size表示窗口大小，stride表示步长。这些参数都被保存在类的属性中。
__getitem__(self, index)：根据给定的索引index获取数据集中的一个样本。
首先，从输入数据data和标签labels中获取完整的记录和对应的标签。然后，使用窗口大小和步长，对记录进行分段，得到具有重叠窗口的分段。这些分段和对应的标签被作为样本的内容返回。
在你提供的具体数据情况下：
data的维度为(2203, 12, 1000)，表示共有2203个记录，每个记录包含12个特征，并且每个特征有1000个时间步。
labels的维度为(2203, 71)，表示每个记录对应了71个类别的标签。
window_size被设置为250，表示每个分段的窗口大小为250个时间步。
7.stride被设置为50，表示每次滑动窗口的步长为50个时间步。
根据提供的数据，该类将会生成具有重叠窗口的分段作为样本，并将对应的标签一起返回。"""
class StepWiseTestset(Dataset):
    
    # Testset for Element-Wise Maximum    元素最大值测试集
    def __init__(self, data, labels, window_size, stride):
        
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        # Get the full record and its corresponding label  # 获取完整记录及其相应标签
        record = self.data[index]
        label = self.labels[index]

        # Create segments with overlapping windows   # 创建具有重叠窗口的分段
        segments = []
        start = 0
        while start + self.window_size <= record.shape[-1]:
            segment = record[:, start : start + self.window_size]
            segments.append(segment)
            start += self.stride

        return segments, label

#%%
'Beautiful Chart for Confusion Matrix混淆矩阵的精美图表'
"""
这段代码定义了一个用于绘制混淆矩阵的函数plot_cm，并指定了使用seaborn风格的Matplotlib图形。
1.cm：混淆矩阵。
2.classes：类别标签。
3.normalize：一个布尔值，指示是否应该对混淆矩阵进行归一化处理。
4.title：图的标题。
5.cmap：颜色映射，指定绘制混淆矩阵时使用的颜色。
函数首先检查是否需要对混淆矩阵进行归一化处理，如果normalize为True，则将混淆矩阵归一化。然后，它打印出混淆矩阵的内容。
接着，函数使用Matplotlib绘制混淆矩阵的图形表示。
其中，每个单元格的颜色表示预测值与真实值之间的关系，颜色越深表示关系越强。
函数还在每个单元格中添加了数字，表示该位置上的值。最后，函数设置图的标题、颜色条、标签等，并显示图形。
这个函数的目的是为了可视化混淆矩阵，以便更直观地理解分类模型的性能。"""
mpl.style.use('seaborn')
def plot_cm(cm, classes, 
           normalize=False,
           title="Confusion Matrix",
           cmap=plt.cm.Purples):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis =1)[:, np.newaxis]
        print("Normalized CM")
    else:
        print('CM without Normalization')
        
    print(cm)
    
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, )
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt),
                horizontalalignment='center',
                color="white" if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

#%%
'Finding Threshold'
"""
这段代码定义了一个函数get_optimal_precision_recall，用于找到最大化 F1 分数的精确度和召回率值。
函数的参数包括：
1.y_true：真实标签的数组。
2.y_score：模型预测的分数数组。
首先初始化了三个空列表opt_precision、opt_recall和opt_threshold，用于存储每个类别的最佳精确度、召回率和阈值。
然后，函数对每个类别进行循环。
对于每个类别，它使用precision_recall_curve函数计算精确度-召回率曲线，并将结果存储在precision、recall和threshold变量中。
接着，函数计算 F1 分数，并通过np.nan_to_num将可能出现的 NaN 值转换为 0，以避免对 F1 分数的计算产生影响。
接下来，函数找到最大 F1 分数对应的索引，并将相应的精确度、召回率和阈值存储在opt_precision、opt_recall和opt_threshold列表中。
最后，函数返回三个列表，分别包含了每个类别的最佳精确度、召回率和阈值。
这个函数的目的是为了在多类别分类问题中，找到最佳的精确度、召回率和阈值，以优化模型的性能。"""
def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], 
                                                              y_score[:, k])
        # Compute f1 score with nan_to_num to avoid nans messing
        _score = np.nan_to_num(2 * precision * recall / (precision + recall))
        # Select threshold that maximize f1 score
        index = np.argmax(_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = threshold[index-1] if index != 0 else threshold[0]-1e-10
        opt_threshold.append(t)
    return opt_precision, opt_recall, opt_threshold
