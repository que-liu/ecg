"""
这是处理所有标签的代码
"""
import torch
import numpy as np
from torch.autograd import Variable

from models.cond_pulse2pulse import CondP2PDiscriminator
from models.cond_pulse2pulse import CondP2PGenerator


from models.cond_wavegan_star import CondWaveGANDiscriminator
from models.cond_wavegan_star import CondWaveGANGenerator

#加载模型和检查点：定义了判别器和生成器模型，并加载了训练好的参数。
# 这是P2P的权重
# chk_path = r"/home/zyw/gans/P2P/p2p/checkpoints/checkpoint_epoch_3000.pth"
# netD = CondP2PDiscriminator(model_size=50, ngpus=1).cuda()
# netG = CondP2PGenerator(model_size=50, ngpus=1).cuda()


#这是WAVEGAN_COND的权重
chk_path = r"/content/ecg/gans/WAVEGAN_COND/all_labels_uncond_star/checkpoints/checkpoint_epoch_3000.pth"
netD = CondWaveGANDiscriminator(model_size=50, ngpus=1).cuda()
netG = CondWaveGANGenerator(model_size=50, ngpus=1).cuda()




def generate_four_leads(tensor):
    leadI = tensor[:,0,:].unsqueeze(1)  #leadI 是原始信号的第一个导联
    leadschest = tensor[:,1:7,:]    #leadschest 是原始信号的胸导联（通常是多个导联）。
    leadavf = tensor[:,7,:].unsqueeze(1)    #leadavf 是原始信号的 aVF 导联


    leadII = (0.5*leadI) + leadavf

    leadIII = -(0.5*leadI) + leadavf
    leadavr = -(0.75*leadI) -(0.5*leadavf)
    leadavl = (0.75*leadI) - (0.5*leadavf)
    #根据心电图信号的计算公式，计算了其他导联（II、III、aVR、aVL），并且将它们重新组合成了一个新的张量 leads12。

    leads12 = torch.cat([leadI, leadII, leadschest, leadIII, leadavr, leadavl, leadavf], dim=1)
    #函数接收生成的信号张量作为输入，然后根据心电图信号的特性，从该张量中提取四个导联的信号。
    return leads12

chkpnt = torch.load(chk_path)

netD.load_state_dict(chkpnt["netD_state_dict"])
netG.load_state_dict(chkpnt["netG_state_dict"])

# x = Variable(torch.randn(32, 8, 1000)).cuda() #准备输入数据：生成了一个大小为 (32, 8, 1000) 的张量 x，代表了输入的随机数据
#载入训练标签
labels_all = np.load(r'/content/ecg/Dataset/dataset/label/1000_valid_labels.npy')
# 定义批大小
batch_size = 32
# 获取总样本数
num_samples = len(labels_all)
# 计算总批次数
num_batches = (num_samples + batch_size - 1) // batch_size
# 定义空列表存储生成的信号
generated_signals = []
# 循环遍历批次
for i in range(num_batches):
    # 计算当前批次的起始索引和结束索引
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, num_samples)

    # 获取当前批次的标签数据
    labels_batch = labels_all[start_idx:end_idx]
    labels_batch = torch.from_numpy(labels_batch).cuda().float()

    # 生成随机输入数据
    x = Variable(torch.randn(labels_batch.size(0), 8, 1000)).cuda()

    # 使用生成器生成输出信号
    out = netG(x, labels_batch)

    # 生成四导联信号
    imputations12_batch = generate_four_leads(out)

    # 将生成的信号添加到列表中
    generated_signals.append(imputations12_batch.cpu().detach().numpy())

# 将生成的信号列表堆叠成一个数组
generated_signals = np.concatenate(generated_signals, axis=0)

# 保存生成的信号为.npy文件
# np.save(r'/home/zyw/gans/P2P/p2p/generated3000_val_signals.npy', generated_signals)
np.save(r'/content/ecg/gans/WAVEGAN_COND/all_labels_uncond_star/checkpoints/generated2970_val_signals.npy', generated_signals)
print("saved!")








