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
#from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchsummary import summary
from torch import autograd

# Model specific
# from data.ecg_data_loader import ECGDataSimple as ecg_data
from models.cond_wavegan_star import CondWaveGANGenerator
from models.cond_wavegan_star import CondWaveGANDiscriminator
from utils.utils import calc_gradient_penalty

torch.manual_seed(0)
np.random.seed(0)
parser = argparse.ArgumentParser()

# Hardware
parser.add_argument("--device_id", type=int, default=0, help="Device ID to run the code")

parser.add_argument("--exp_name", type=str, default="all_labels_uncond_star", 
                    help="A name to the experiment which is used to save checkpoitns and tensorboard output")


#==============================
# Directory and file handling
#==============================
parser.add_argument("--out_dir", 
                    default="WAVEGAN_COND",
                    help="Main output dierectory")

parser.add_argument("--tensorboard_dir", 
                    default="WAVEGAN_COND",
                    help="Folder to save output of tensorboard")

#======================
# Hyper parameters
#======================
parser.add_argument("--bs", type=int, default=32, help="Mini batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")

parser.add_argument("--num_epochs", type=int, default=3000, help="number of epochs of training")
parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch in retraining")
parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs used in models")
parser.add_argument("--checkpoint_interval", type=int, default=30, help="Interval to save checkpoint models")  #间隔保存模型

# Checkpoint path to retrain or test models    重新训练或测试模型的检查点路径
parser.add_argument("--checkpoint_path", default=r"/content/ecg/gans/WAVEGAN_COND/all_labels_uncond_star/checkpoints/checkpoint_epoch_2970.pth", help="Check point path to retrain or test models")

parser.add_argument('-ms', '--model_size', type=int, default=50,
                        help='Model size parameter used in WaveGAN')
parser.add_argument('--lmbda', default=10.0, help="Gradient penalty regularization factor")

# Action handling 
parser.add_argument("--action", 
                    type=str, 
                    default='train',
                    help="Select an action to run", 
                    choices=["train", "retrain", "inference", "check"])


opt = parser.parse_args()
print(opt)

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
os.makedirs(opt.out_dir, exist_ok=True)


# make subfolder in the output folder 
checkpoint_dir = os.path.join(opt.out_dir, opt.exp_name + "/checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# make tensorboard subdirectory for the experiment
tensorboard_exp_dir = os.path.join(opt.tensorboard_dir, opt.exp_name)
os.makedirs( tensorboard_exp_dir, exist_ok=True)

#==========================================
# Prepare Data
#==========================================



def prepare_data():
    dataset = np.load(r"/content/ecg/Dataset/dataset/train_ptbxl_1000.npy")
   
    index_8 = torch.tensor([0,2,3,4,5,6,7,11])
    index_4 = torch.tensor([1,8,9,10])
    
    dataset = torch.index_select(torch.from_numpy(dataset), 1, index_8).float()
    labels = np.load(r"/content/ecg/Dataset/label/1000_train_labels.npy")
    
    data = []
    for signal, label in zip(dataset, labels):
        data.append([signal, label])
        
    dataloader = torch.utils.data.DataLoader(data, batch_size=opt.bs, shuffle=True, drop_last=True)
 
    return dataloader



#===============================================
# Prepare models
#===============================================
def prepare_model():
    netG = CondWaveGANGenerator(model_size=opt.model_size, ngpus=opt.ngpus, upsample=True)
    netD = CondWaveGANDiscriminator(model_size=opt.model_size, ngpus=opt.ngpus)

    netG = netG.to(device)
    netD = netD.to(device)

    return netG, netD

#====================================
# Run training process
#====================================
def run_train():
    netG, netD = prepare_model()
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    dataloaders = prepare_data() 
    train(netG, netD, optimizerG, optimizerD, dataloaders)

    
    
def train(netG, netD, optimizerG, optimizerD, dataloader):

    for epoch in tqdm(range(opt.start_epoch + 1, opt.start_epoch + opt.num_epochs + 1)):
        
        print('\n')
        print('This is the epoch number', epoch)
        print('\n')
        
        len_dataloader = len(dataloader)

        train_G_flag = False
        D_cost_train_epoch = []
        D_wass_train_epoch = []
        G_cost_epoch = []
    
        for i, sample in tqdm(enumerate(dataloader, 0)):
            
            labels = sample[1]
            sample = sample[0]
            
            sample = {'ecg_signals': sample}

            if (i+1) % 5 == 0:
                train_G_flag = True


            # Set Discriminator parameters to require gradients.
            #print(train_G_flag)
            for p in netD.parameters():
                p.requires_grad = True

            #one = torch.Tensor([1]).float()
            one = torch.tensor(1, dtype=torch.float)
            neg_one = one * -1

            one = one.to(device)
            neg_one = neg_one.to(device)

            #############################
            # (1) Train Discriminator
            #############################

            real_ecgs = sample['ecg_signals'].to(device)
            
            #print("real ecgs shape", real_ecgs.shape)
            b_size = real_ecgs.size(0)

            netD.zero_grad()


            # Noise
            noise = torch.Tensor(b_size, 8, 1000).uniform_(-1, 1)
            noise = noise.to(device)
            noise_Var = Variable(noise, requires_grad=False)


            # a) compute loss contribution from real training data
            D_real = netD(real_ecgs)
            D_real = D_real.mean()  # avg loss
            D_real.backward(neg_one)  # loss * -1

            # b) compute loss contribution from generated data, then backprop.
            fake = autograd.Variable(netG(noise_Var, labels.cuda()).data)
            D_fake = netD(fake)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # c) compute gradient penalty and backprop
            gradient_penalty = calc_gradient_penalty(netD, real_ecgs,
                                                    fake.data, b_size, opt.lmbda,
                                                    use_cuda=True)
            gradient_penalty.backward(one)

            # Compute cost * Wassertein loss..
            D_cost_train = D_fake - D_real + gradient_penalty
            D_wass_train = D_real - D_fake

            # Update gradient of discriminator.
            optimizerD.step()

            D_cost_train_cpu = D_cost_train.data.cpu()
            D_wass_train_cpu = D_wass_train.data.cpu()


            D_cost_train_epoch.append(D_cost_train_cpu)
            D_wass_train_epoch.append(D_wass_train_cpu)


            #############################
            # (3) Train Generator
            #############################
            if train_G_flag:
                # Prevent discriminator update.
                for p in netD.parameters():
                    p.requires_grad = False

                # Reset generator gradients
                netG.zero_grad()

                # Noise
                noise = torch.Tensor(b_size, 8, 1000).uniform_(-1, 1)
                
                noise = noise.to(device)
                noise_Var = Variable(noise, requires_grad=False)

                fake = netG(noise_Var, labels.cuda())
                G = netD(fake)
                G = G.mean()

                # Update gradients.
                G.backward(neg_one)
                G_cost = -G

                optimizerG.step()

                # Record costs
                #if cuda:
                G_cost_cpu = G_cost.data.cpu()
                G_cost_epoch.append(G_cost_cpu)
                train_G_flag =False

        D_cost_train_epoch_avg = sum(D_cost_train_epoch) / float(len(D_cost_train_epoch))
        D_wass_train_epoch_avg = sum(D_wass_train_epoch) / float(len(D_wass_train_epoch))
        G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

        print("Epochs:{}\t\tD_cost:{}\t\t D_wass:{}\t\tG_cost:{}".format(
                    epoch, D_cost_train_epoch_avg, D_wass_train_epoch_avg, G_cost_epoch_avg))

         # Save model
        if epoch % opt.checkpoint_interval == 0:
            save_model(netG, netD, optimizerG, optimizerD, epoch)


#=====================================
# Save models
#=====================================
def save_model(netG, netD, optimizerG, optimizerD,  epoch):
   
    # check_point_name = "checkpoint_epoch:{}.pt".format(epoch) # get code file name and make a name
    check_point_name = "checkpoint_epoch_{}.pth".format(epoch)
    check_point_path = os.path.join(checkpoint_dir, check_point_name)
    # save torch model
    torch.save({
        "epoch": epoch,
        "netG_state_dict": netG.state_dict(),
        "netD_state_dict": netD.state_dict(),
        "optimizerG_state_dict": optimizerG.state_dict(),
        "optimizerD_state_dict": optimizerD.state_dict()}, 
      check_point_path)

#====================================
# Re-train process
#====================================
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

    # setup start epoch to checkpoint epoch
    opt.__setattr__("start_epoch", chkpnt["epoch"])

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


    dataloaders = prepare_data()
    train(netG, netD, optimizerG, optimizerD, dataloaders)
    

#=====================================
# Check model
#====================================
def check_model_graph():
    netG, netD = prepare_model()
    print(netG)
    netG = netG.to(device)
    netD = netD.to(device)

    summary(netG, (8, 1000))
    summary(netD, (8, 1000))


if __name__ == "__main__":

    data_loaders = prepare_data()
    print(vars(opt))
    print("Test OK")
    """
{'device_id': 0, 'exp_name': 'all_labels_uncond_star', 'out_dir': 'WAVEGAN_COND', '
tensorboard_dir': 'WAVEGAN_COND', 'bs': 32, 'lr': 0.0001, 'b1': 0.5, 'b2': 0.9, 'num_epochs': 3000, 'start_epoch': 0, 
'ngpus': 1, 'checkpoint_interval': 20, 'checkpoint_path': 'checkpoint_epoch:3000.pt', 'model_size': 50, 'lmbda': 10.0, 
'action': 'train'}"""
    """
    Namespace(checkpoint_interval=30, device_id=0, exp_name='all_labels_uncond_star', lmbda=1) 
    checkpoint_interval': 30,"""
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
        print("Inference process is strted..!")
        pass
    elif opt.action == "check":
        check_model_graph()
        print("Check pass")
#D_cost:-19.345226287841797		 D_wass:25.13629150390625		G_cost:2.7651333808898926
#30		D_cost:-19.485767364501953		 D_wass:25.396379470825195		G_cost:0.8246577382087708
#60		D_cost:-19.566295623779297		 D_wass:25.5543270111084		G_cost:-0.4103688895702362
#90		D_cost:-19.562517166137695		 D_wass:25.55568504333496		G_cost:0.4171428382396698
#120		D_cost:-19.625762939453125		 D_wass:25.610380172729492		G_cost:0.8139731884002686
#150		D_cost:-19.129636764526367		 D_wass:24.950321197509766		G_cost:-0.0530712828040123

#400		D_cost:-17.15264129638672		 D_wass:22.101499557495117		G_cost:-0.7317769527435303
#420		D_cost:-16.893972396850586		 D_wass:21.731733322143555		G_cost:-1.1545356512069702
#440		D_cost:-16.83624839782715		 D_wass:21.65986442565918		G_cost:-0.9576849341392517
#450		D_cost:-16.618318557739258		 D_wass:21.33279800415039		G_cost:-1.3337416648864746

#700		D_cost:-15.778508186340332		 D_wass:20.182476043701172		G_cost:-1.866734266281128
#720		D_cost:-15.592325210571289		 D_wass:19.924776077270508		G_cost:-1.695008397102356
#750		D_cost:-15.787338256835938		 D_wass:20.187965393066406		G_cost:-1.7898238897323608
#780		D_cost:-15.862920761108398		 D_wass:20.28862953186035		G_cost:-2.0864877700805664

#1080		D_cost:-15.636080741882324		 D_wass:19.979211807250977		G_cost:-1.8797014951705933
#1100		D_cost:-15.502947807312012		 D_wass:19.801546096801758		G_cost:-1.76531183719635
#1120		D_cost:-15.566827774047852		 D_wass:19.87151336669922		G_cost:-2.4212660789489746
#1140		D_cost:-15.59517765045166		 D_wass:19.926576614379883		G_cost:-2.4513094425201416

#1270		D_cost:-15.583966255187988		 D_wass:19.909337997436523		G_cost:-2.2097296714782715
#1290		D_cost:-15.5753812789917		 D_wass:19.873361587524414		G_cost:-2.245175361633301
#1340		D_cost:-15.608539581298828		 D_wass:19.932172775268555		G_cost:-2.6713624000549316
#1370		D_cost:-15.463577270507812		 D_wass:19.744443893432617		G_cost:-2.8385331630706787

#1440		D_cost:-15.289119720458984		 D_wass:19.506465911865234		G_cost:-2.9530797004699707
#1470		D_cost:-15.426529884338379		 D_wass:19.702987670898438		G_cost:-2.881284713745117

#1760		D_cost:-14.906375885009766		 D_wass:18.98455047607422		G_cost:-2.5138208866119385
#1790		D_cost:-15.071264266967773		 D_wass:19.24585723876953		G_cost:-2.239466905593872
#1820		D_cost:-14.911528587341309		 D_wass:18.997657775878906		G_cost:-2.58920955657959


#2050		D_cost:-14.873930931091309		 D_wass:18.958709716796875		G_cost:-2.617131233215332
#2080		D_cost:-14.965527534484863		 D_wass:19.083480834960938		G_cost:-2.8070223331451416
#2100		D_cost:-14.96296501159668		 D_wass:19.110410690307617		G_cost:-2.4394450187683105
#2150		D_cost:-14.942900657653809		 D_wass:19.061416625976562		G_cost:-2.443192958831787


#2460		D_cost:-14.683116912841797		 D_wass:18.69584846496582		G_cost:-1.9019975662231445
#2490		D_cost:-14.595438003540039		 D_wass:18.584856033325195		G_cost:-2.140486717224121
#2520		D_cost:-14.597163200378418		 D_wass:18.577672958374023		G_cost:-2.468667507171631
#2550		D_cost:-14.51351261138916		 D_wass:18.469892501831055		G_cost:-2.708801507949829
#2580		D_cost:-14.631556510925293		 D_wass:18.62685203552246		G_cost:-2.957465171813965
#2610		D_cost:-14.607393264770508		 D_wass:18.607221603393555		G_cost:-3.291212558746338
#2640		D_cost:-14.701621055603027		 D_wass:18.722984313964844		G_cost:-3.359705924987793
#2670		D_cost:-14.617924690246582		 D_wass:18.611286163330078		G_cost:-3.512998342514038
#2700		D_cost:-14.730867385864258		 D_wass:18.76611328125		G_cost:-3.3208518028259277
#2750		D_cost:-14.677541732788086		 D_wass:18.715953826904297		G_cost:-2.8344175815582275




#
#2940		D_cost:-14.561347007751465		 D_wass:18.533227920532227		G_cost:-2.9974660873413086
#2970		D_cost:-14.426033020019531		 D_wass:18.330123901367188		G_cost:-2.6837098598480225
#3000		D_cost:-14.442301750183105		 D_wass:18.38184356689453		G_cost:-3.5500519275665283

#2997		D_cost:-14.440848350524902		 D_wass:18.38861846923828		G_cost:-3.310657501220703
#2998		D_cost:-14.431544303894043		 D_wass:18.35636329650879		G_cost:-3.239029884338379
#2999		D_cost:-14.4209623336792		 D_wass:18.340084075927734		G_cost:-3.5397136211395264
