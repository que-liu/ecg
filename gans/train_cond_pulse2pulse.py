import argparse
import os
from tqdm import tqdm
import numpy as np

#Pytorch
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
# from torchvision import models, transforms
# from torchvision.utils import save_image
from torch.autograd import Variable
from torchsummary import summary
from torch import autograd

# Model specific

from models.cond_pulse2pulse import CondP2PGenerator 
from models.cond_pulse2pulse import CondP2PDiscriminator
from utils.utils import calc_gradient_penalty

torch.manual_seed(0)
np.random.seed(0)#随机种子设置：设置了PyTorch和NumPy的随机种子，以确保在每次运行代码时得到相同的随机结果。这对于实验的可重复性是很重要的。
parser = argparse.ArgumentParser()#命令行参数解析器,创建了一个ArgumentParser对象，用于处理命令行参数
#使用Python argparse库的命令行参数解析器。用于训练、重新训练、推断或检查模型，并且支持许多超参数以及文件和目录的处理。

# Hardware硬件和实验名称参数
parser.add_argument("--device_id", type=int, default=0, help="Device ID to run the code")

parser.add_argument("--exp_name", type=str, default="p2p", 
                    help="A name to the experiment which is used to save checkpoitns and tensorboard output")
#实验的名称，用于保存检查点和tensorboard输出


#==============================
# Directory and file handling 目录和文件处理参数,定义了输出目录和保存TensorBoard输出的目录
#==============================

parser.add_argument("--out_dir", 
                    default="P2P",
                    help="Main output dierectory")

parser.add_argument("--tensorboard_dir", 
                    default="P2P",
                    help="Folder to save output of tensorboard")#用于保存tensorboard输出的文件夹


#======================
# Hyper parameters 超参数       模型训练的超参数，如批量大小、学习率等。
#======================
parser.add_argument("--bs", type=int, default=32, help="Mini batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")

parser.add_argument("--num_epochs", type=int, default=3000, help="number of epochs of training")
parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch in retraining")
parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs used in models")
parser.add_argument("--checkpoint_interval", type=int, default=20, help="Interval to save checkpoint models") #保存检查点模型的间隔时间

# Checkpoint path to retrain or test models  重新训练或测试模型的检查点路径
parser.add_argument("--checkpoint_path", default=r"/content/ecg/gans/P2P/p2p/checkpoints/checkpoint_epoch_180.pth", help="Check point path to retrain or test models")

parser.add_argument('-ms', '--model_size', type=int, default=50,
                        help='Model size parameter used in WaveGAN')
parser.add_argument('--lmbda', default=10.0, help="Gradient penalty regularization factor")

# Action handling  action处理
parser.add_argument("--action",
                    type=str,
                    default='retrain',
                    help="Select an action to run",
                    choices=["train", "retrain", "inference", "check"])
# parser.add_argument("--action",
#                     type=str,
#                     default='',
#                     help="Select an action to run",
#                     choices=["train", "retrain", "inference", "check"])


opt = parser.parse_args()
print(opt)
"""
6.设备处理和输出目录创建：
   opt = parser.parse_args()
   torch.cuda.set_device(opt.device_id)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   os.makedirs(opt.out_dir, exist_ok=True)

这里解析了命令行参数，设置了GPU设备，并在需要时创建了输出目录。
"""
#==========================================
# Device handling
#==========================================
torch.cuda.set_device(opt.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device=", device)

#===========================================
# Folder handling
#===========================================

#make output folder if not exist
os.makedirs(opt.out_dir, exist_ok=True)    #P2P


# make subfolder in the output folder 在输出文件夹中创建子文件夹
checkpoint_dir = os.path.join(opt.out_dir, opt.exp_name + "/checkpoints")   #out+p2p
os.makedirs(checkpoint_dir, exist_ok=True)

# make tensorboard subdirectory for the experiment  为实验制作tensorboard子目录
tensorboard_exp_dir = os.path.join(opt.tensorboard_dir, opt.exp_name)
os.makedirs( tensorboard_exp_dir, exist_ok=True)


#==========================================
# Prepare Data
#==========================================

def prepare_data():#定义了一个名为prepare_data的函数，用于准备数据集并创建数据加载器（dataloader）。
    dataset = np.load(r"/content/ecg/gans/dataset/data/train_ptbxl_1000.npy")
   
    
    index_8 = torch.tensor([0,2,3,4,5,6,7,11])
    index_4 = torch.tensor([1,8,9,10])  #定义了两个索引张量，用于从数据集中选择特定的信号。index_8选择了索引为0、2、3、4、5、6、7、11的信号，index_4选择了索引为1、8、9、10的信号
    
    dataset = torch.index_select(torch.from_numpy(dataset), 1, index_8).float() #PyTorch的index_select函数根据索引index_8选择数据集中的特定信号，并将结果转换为浮点张量。
    labels = np.load(r"/content/ecg/gans/dataset/label/1000_train_labels.npy")
    
    data = []
    
    for signal, label in zip(dataset, labels): #这一行代码使用zip函数将数据集中的信号和标签一一对应起来进行遍历。
        data.append([signal, label])
    
    dataloader = torch.utils.data.DataLoader(data, batch_size=opt.bs, shuffle=True)    
    return dataloader

#===============================================
# Prepare models
#===============================================
def prepare_model():  #用于准备生成器（Generator）和判别器（Discriminator）模型。并将它们移动到指定的设备上，然后返回这两个模型。
    netG = CondP2PGenerator(model_size=opt.model_size, ngpus=opt.ngpus, upsample=True)
    netD = CondP2PDiscriminator(model_size=opt.model_size, ngpus=opt.ngpus)

    netG = netG.to(device)
    netD = netD.to(device)

    return netG, netD

#====================================
# Run training process
#====================================
"""
实现了一个训练过程，使用了生成对抗网络（GAN）。以下是对这段代码的中文解释：
2.初始化优化器： 使用 Adam 优化器初始化生成器和判别器的参数。lr 参数是学习率，而 betas 参数是 Adam 优化器的动量项。
3.准备数据加载器： prepare_data() 函数返回用于训练的数据加载器，其中包含训练数据的迭代器。
4.调用训练函数： train() 函数执行模型的训练过程，接收生成器、判别器、生成器优化器、判别器优化器以及数据加载器作为输入。
这样的结构是典型的GAN 训练流程，其中生成器和判别器通过博弈的方式相互学习，从而提高生成器生成真实样本的能力，同时判别器更好地区分真实和生成的样本。
"""
def run_train():
    netG, netD = prepare_model()  ## 准备生成器和判别器模型
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    dataloaders = prepare_data() 
    train(netG, netD, optimizerG, optimizerD, dataloaders)

def train(netG, netD, optimizerG, optimizerD, dataloader):
    """通过遍历数据加载器中的每个批次，对判别器和生成器进行了训练。在每个批次中，首先训练判别器，然后根据需要训练生成器。最后，计算并打印每个 epoch 的平均损失。"""

    for epoch in tqdm(range(opt.start_epoch + 1, opt.start_epoch + opt.num_epochs + 1)):

        print('\n')
        print('This is the epoch number', epoch)
        print('\n')
        
        len_dataloader = len(dataloader)
        #print("Length of Dataloader:", len_dataloader)

        train_G_flag = False     # 初始化标志变量，用于控制是否训练生成器
        D_cost_train_epoch = []   # 存储当前 epoch 中判别器的损失和 Wassertein 距离
        D_wass_train_epoch = []
        G_cost_epoch = []        # 存储当前 epoch 中生成器的损失

        for i, sample in tqdm(enumerate(dataloader, 0)):  #是一个完整的训练循环，其中包含了对生成对抗网络（GAN）中判别器和生成器的训练过程。

            labels = sample[1]
            sample = sample[0]
            
            sample = {'ecg_signals': sample} # 将样本数据封装为字典

            if (i+1) % 5 == 0:
                train_G_flag = True# 每隔5个批次，设置标志以训练生成器



            # Set Discriminator parameters to require gradients.设置判别器参数需要梯度
            #print(train_G_flag)
            for p in netD.parameters():
                p.requires_grad = True

            #one = torch.Tensor([1]).float()
            one = torch.tensor(1, dtype=torch.float)    # 创建张量1
            neg_one = one * -1                    # 创建张量-1

            one = one.to(device)
            neg_one = neg_one.to(device)

            #############################
            # (1) Train Discriminator        训练判别器
            #############################

            real_ecgs = sample['ecg_signals'].to(device)# 获取真实心电图数据

            #print("real ecgs shape", real_ecgs.shape)
            b_size = real_ecgs.size(0)# 获取批次大小

            netD.zero_grad()# 判别器梯度清零


            # Noise噪声
            noise = torch.Tensor(b_size, 8, 1000).uniform_(-1, 1)   # 生成均匀分布的噪声
            noise = noise.to(device)
            noise_Var = Variable(noise, requires_grad=False)   # 将噪声转换为变量

            # real_data_Var = numpy_to_var(next(train_iter)['X'], cuda)

            # a) compute loss contribution from real training data     计算真实训练数据的损失

            D_real = netD(real_ecgs)
            D_real = D_real.mean()  # avg loss平均损失
            D_real.backward(neg_one)  # loss * -1反向传播损失 * -1

            # b) compute loss contribution from generated data, then backprop.计算生成数据的损失，并进行反向传播
            fake = autograd.Variable(netG(noise_Var, labels.cuda()).data)# 生成假数据
            D_fake = netD(fake)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # c) compute gradient penalty and backprop   计算梯度惩罚并进行反向传播
            gradient_penalty = calc_gradient_penalty(netD, real_ecgs,
                                                    fake.data, b_size, opt.lmbda,
                                                    use_cuda=True)
            gradient_penalty.backward(one)

            # Compute cost * Wassertein loss..# 计算总损失和Wasserstein损失
            D_cost_train = D_fake - D_real + gradient_penalty
            D_wass_train = D_real - D_fake

            # Update gradient of discriminator. # 更新判别器梯度
            optimizerD.step()

            D_cost_train_cpu = D_cost_train.data.cpu()
            D_wass_train_cpu = D_wass_train.data.cpu()


            D_cost_train_epoch.append(D_cost_train_cpu)
            D_wass_train_epoch.append(D_wass_train_cpu)


            #############################
            # (3) Train Generator 训练生成器
            #############################

            if train_G_flag:
                # Prevent discriminator update.阻止判别器更新
                for p in netD.parameters():
                    p.requires_grad = False

                # Reset generator gradients    重置生成器梯度
                netG.zero_grad()

                # Noise
                noise = torch.Tensor(b_size, 8, 1000).uniform_(-1, 1)# 重新生成噪声
                
                noise = noise.to(device)
                noise_Var = Variable(noise, requires_grad=False) # 将噪声转换为变量

                fake = netG(noise_Var, labels.cuda())     ################ pass labels here!# 生成假数据
                G = netD(fake)
                G = G.mean()

                # Update gradients. # 更新生成器梯度
                G.backward(neg_one)
                G_cost = -G

                optimizerG.step()

                # Record costs# 记录生成器损失
                #if cuda:
                G_cost_cpu = G_cost.data.cpu()
                #print("g_cost=",G_cost_cpu)
                G_cost_epoch.append(G_cost_cpu)
                #print("Epoch{} - {}_G_cost_cpu:{}".format(epoch, i, G_cost_cpu))
                #G_cost_epoch.append(G_cost_cpu.data.numpy())
                train_G_flag =False     # 重置生成器训练标志
        #  # 计算并打印平均损失
        D_cost_train_epoch_avg = sum(D_cost_train_epoch) / float(len(D_cost_train_epoch))
        D_wass_train_epoch_avg = sum(D_wass_train_epoch) / float(len(D_wass_train_epoch))
        G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

        print("Epochs:{}\t\tD_cost:{}\t\t D_wass:{}\t\tG_cost:{}".format(
                    epoch, D_cost_train_epoch_avg, D_wass_train_epoch_avg, G_cost_epoch_avg))

         # Save model
        if epoch % opt.checkpoint_interval == 0:
            save_model(netG, netD, optimizerG, optimizerD, epoch)
            print("成功保存模型epoch",epoch)


#=====================================
# Save models   保存训练过程中的模型和优化器参数的函数
#=====================================
def save_model(netG, netD, optimizerG, optimizerD,  epoch):
   
    #check_point_name = "checkpoint_epoch:{}.pth".format(epoch) # get code file name and make a name
    check_point_name = "checkpoint_epoch_{}.pth".format(epoch)
    check_point_path = os.path.join(checkpoint_dir, check_point_name)
    # save torch model    保存 PyTorch 模型和优化器的状态字典到指定路径
    torch.save({
        "epoch": epoch,
        "netG_state_dict": netG.state_dict(),    #保存生成器模型的状态字典
        "netD_state_dict": netD.state_dict(),
        "optimizerG_state_dict": optimizerG.state_dict(),# 保存生成器优化器的状态字典
        "optimizerD_state_dict": optimizerD.state_dict()},
      check_point_path)

    
#====================================
# Re-train process重新训练流程
#====================================
"""
1.打印提示信息，表示重新训练开始。
2.调用 prepare_model 函数准备生成器（netG）和判别器（netD）模型。
3.使用 torch.load 加载预训练模型的检查点。
4.将生成器和判别器的状态字典加载到相应的模型中。
5.将模型移到目标设备上（例如 GPU）。
6.打印加载模型的信息。
7.将起始 epoch 设置为检查点的 epoch。
8.设置生成器和判别器的 Adam 优化器。
9.调用 prepare_data 函数准备数据加载器。
10.调用 train 函数开始重新训练过程。
目的是在预训练模型的基础上进行额外的训练，以适应新的数据或进一步优化模型。"""
def run_retrain():
    print("run retrain started........................")
    netG, netD = prepare_model()

    # loading checkpoing
    chkpnt = torch.load(opt.checkpoint_path, map_location="cpu")

    netG.load_state_dict(chkpnt["netG_state_dict"])
    netD.load_state_dict(chkpnt["netD_state_dict"])

    netG = netG.to(device)
    netD = netD.to(device)

    print("model loaded from checkpoint=", opt.checkpoint_path)

    # setup start epoch to checkpoint epoch  # 将起始 epoch 设置为检查点的 epoch
    opt.__setattr__("start_epoch", chkpnt["epoch"])

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


    dataloaders = prepare_data()
    train(netG, netD, optimizerG, optimizerD, dataloaders)
    

#=====================================
# Check model     定义 check_model_graph 的函数，其目的是检查生成器和判别器模型的图形结构
#====================================
def check_model_graph():
    netG, netD = prepare_model()
    print(netG)   #打印生成器模型 netG 的结构信息，通常会显示模型的各个层和参数。
    netG = netG.to(device)
    netD = netD.to(device)

    summary(netG, (8, 1000))   #使用torchsummary库中的 summary 函数，打印生成器模型 netG 的摘要信息。摘要信息包括模型的层次结构、每一层的输出形状以及模型的参数数量等。
    summary(netD, (8, 1000))


if __name__ == "__main__":

    data_loaders = prepare_data()
    #vars() 是一个内置函数，用于返回对象的__dict__属性。在Python中，每个对象都有一个__dict__属性，它是一个字典，包含对象的属性和它们的值。
    #在这种情况下，opt 是一个由 argparse.ArgumentParser 解析命令行参数后得到的对象。当你调用 vars(opt) 时，它将返回一个包含 opt 对象的所有属性和它们的值的字典。

    print(vars(opt))  #打印出解析后的命令行参数的字典表示，显示了用户设置的配置。
    print("Test OK")
    """
{'device_id': 0, 'exp_name': 'p2p', 'out_dir': 'P2P', 'tensorboard_dir': 'P2P', 'bs': 32, 'lr': 0.0001, 'b1': 0.5, 'b2': 0.9, 'num_epochs': 3000, 'start_epoch': 0, 
'ngpus': 1, 'checkpoint_interval': 20, 'checkpoint_path': 'checkpoint_epoch:3000.pt', 'model_size': 50, 'lmbda': 10.0, 'action': 'train'}
"""
    # Train or retrain or inference
    if opt.action == "train":
        print("Training process is strted..!")
        run_train()
        pass
    elif opt.action == "retrain":
        print("Retrainning process is strted..!")
        run_retrain()
        pass
    elif opt.action == "inference":
        print("Inference process is strted..!")  #执行推断过程的代码（在此处是没有提供的）。
        pass
    elif opt.action == "check":
        check_model_graph()
        print("Check pass")
#2920		D_cost:-8.1781005859375		 D_wass:10.043190002441406		G_cost:-7.602492809295654
#2940		D_cost:-8.172077178955078		 D_wass:10.080268859863281		G_cost:-7.820872783660889
#2960		D_cost:-8.16994857788086		 D_wass:10.030049324035645		G_cost:-7.374550819396973
#2980		D_cost:-8.235926628112793		 D_wass:10.155274391174316		G_cost:-7.330538749694824
#3000		D_cost:-8.242384910583496		 D_wass:10.12869930267334		G_cost:-7.047961711883545

#3040		D_cost:-8.220986366271973		 D_wass:10.102450370788574		G_cost:-6.368056774139404
#3060		D_cost:-8.217846870422363		 D_wass:10.124895095825195		G_cost:-6.571524143218994
#3080		D_cost:-8.127889633178711		 D_wass:10.008098602294922		G_cost:-7.298390865325928
#3100		D_cost:-8.023797988891602		 D_wass:9.83349895477295		G_cost:-6.228384494781494
#3120		D_cost:-8.089190483093262		 D_wass:9.947507858276367		G_cost:-6.420660972595215
#3140		D_cost:-8.18470287322998		 D_wass:10.100556373596191		G_cost:-6.65012788772583
#3160		D_cost:-8.141042709350586		 D_wass:10.084186553955078		G_cost:-6.3547844886779785
#3180		D_cost:-8.113761901855469		 D_wass:10.001199722290039		G_cost:-5.91156530380249
