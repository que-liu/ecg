"""
初始化模型: 类似于训练之前初始化模型架构，确保配置完全匹配。
加载保存的权重: 从保存的检查点文件中加载权重。
在新数据集上评估: 使用加载的模型在新数据集上进行评估。"""
from datetime import datetime
import math
from sklearn.metrics import f1_score, confusion_matrix
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from params import *
from preprocessing import *
from utils import *
from xresnet1d import *
import torch

seed =123  # 设定一个整数种子值

# 设置随机种子
np.random.seed(seed)
torch.manual_seed(seed)  #在add_10000时候设置的
model = xresnet1d50(model_drop_r=model_dropout, original_f_number=False, fc_drop=fc_drop)
model.to(device)
# 加载保存的权重
#原始数据训练的权重          更新为正确的路径
# checkpoint_path = r"D:\xResnet-main\xResnet1d-main\results\raw\model_weight_val100.pt"
checkpoint_path = r"/content/ecg/xResnet1d-main/results/raw/model_weight_val81.pt"

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
#验证数据
val_x = np.load(r"/content/ecg/Datasets/dataset/val_ptbxl_1000.npy")
val_y = np.load(r"/content/ecg/Datasets/label/1000_valid_labels.npy")
new_dataset =  Dataset(val_x, val_y, int(w_size * model_sr))  # 使用适当的参数初始化您的新数据集
new_data_loader = DataLoader(new_dataset, batch_size=64, shuffle=True)
loss_fn = torch.nn.BCEWithLogitsLoss()
# 在新数据集上评估
model.eval()
total_loss = 0.0
total_f1 = 0.0
num_batches = 0
num_classes = 71  #总类别数
class_f1_scores = np.zeros(num_classes)  # 存储每个类的 F1 分数
class_counts = np.zeros(num_classes)  # 存储每个类别的出现次数，用于计算平均值
y_true = []
y_scores = []
with torch.no_grad():
    for x_new, y_new in new_data_loader:
        x_new, y_new = x_new.to(device, torch.float32), y_new.to(device, torch.float32)
        y_hat_new = model(x_new)
        loss_new = loss_fn(y_hat_new, y_new)
        total_loss += loss_new.item()

        y_hat_new_prob = out_activation(y_hat_new).cpu().detach().numpy()
        y_true.extend(y_new.cpu().numpy().astype('int32'))
        y_scores.extend(y_hat_new_prob)

        y_hat_new_binary = (y_hat_new_prob >= 0.5).astype(int)
        # y_hat_new_binary = (out_activation(y_hat_new).cpu().detach().numpy() >= .5).astype(int)
        f1_new = f1_score(y_new.cpu().numpy().astype('int32'), y_hat_new_binary, average='samples', zero_division=0)
        # 对于每个类别单独计算 F1 分数
        y_new_cpu = y_new.cpu().numpy().astype(int)
        for i in range(num_classes):
            class_y_true = y_new_cpu[:, i]
            class_y_pred = y_hat_new_binary[:, i]
            class_f1 = f1_score(class_y_true, class_y_pred, zero_division=0)
            class_f1_scores[i] += class_f1
            class_counts[i] += 1
        total_f1 += f1_new
        num_batches += 1
average_loss = total_loss / num_batches
average_f1 = total_f1 / num_batches
print(f"新数据集上的平均损失：{average_loss}")
print(f"新数据集上的平均 F1 分数：{average_f1}")

# 计算每个类别的平均 F1 分数
average_f1_scores = class_f1_scores / class_counts
for i in range(num_classes):
    print(f"疾病 {i+1} 的平均 F1 分数：{average_f1_scores[i]}")
# 计算 AUROC
y_true = np.array(y_true)
y_scores = np.array(y_scores)
auroc = roc_auc_score(y_true, y_scores, average='macro')  # 使用'macro'平均方式计算 AUROC
print(f"新数据集上的平均 F1 分数：{average_f1}")
print(f"新数据集上的 AUROC：{auroc}")

