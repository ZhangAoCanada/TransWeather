import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data_functions import TrainData
from val_data_functions import ValData
from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
import os
import numpy as np
import random 
from torch.utils.tensorboard import SummaryWriter

from transweather_model import Transweather

from train_psnrloss import PSNRLoss
import pytorch_ssim

from utils import calc_psnr, calc_ssim

### NOTE: add GD path ###
# import sys
# sys.path.append('/content/drive/MyDrive/train/data')

### NOTE: backend agg ###
# plt.switch_backend('agg')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[512, 512], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=7, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
# parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.01, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default="ckpt", type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=1000, type=int)
parser.add_argument('-logdir', help='for tensorboard', default="TeacherTry12.5_combine", type=str)
# parser.add_argument('-logdir', help='for tensorboard', default="new_try12", type=str)

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
tensorboard_logdir = args.logdir


#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # If GPU is available, change to GPU
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, lambda_loss))


##################### NOTE: Change the path to the dataset #####################
# train_data_dir = "/home/za/DATASET/DATA_20220531/train"
# validate_data_dir = "/home/za/DATASET/DATA_20220531/validate"
# test_data_dir = "/home/zhangao/za/DATA_20220531/test_specific"
train_data_dir = "/home/za/DATASET/DATA_20220617/train"
validate_data_dir = "/home/za/DATASET/DATA_20220617/validate"
test_data_dir = "/home/za/DATASET/DATA_20220617/test_specific"
rain_L_dir = "rain_L"
rain_H_dir = "rain_H"
gt_dir = "gt"

# train_data_dir = "/dataset/public/raindrop/train"
# validate_data_dir = "/dataset/public/raindrop/test"
# rain_L_dir = "data"
# rain_H_dir = None
# gt_dir = "gt"

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = Transweather()


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False

modelbest_path = "./{}/latest".format(exp_name)

# --- Load the network weight --- #
if os.path.exists('./{}/'.format(exp_name))==False:     
    os.mkdir('./{}/'.format(exp_name))  

if os.path.exists(modelbest_path):
    resume_state = torch.load(modelbest_path)
    net.load_state_dict(resume_state["state_dict"])
    optimizer.load_state_dict(resume_state["optimizer"])
    epoch_start = resume_state["epoch"]
    print("----- latest trained loaded -----")
elif os.path.exists("./ckpt/pretrained"):
    # net.load_state_dict(torch.load('./{}/pretrained_on_data2070'.format(exp_name)))
    # print("----- pre-trained with DATA2070images loaded -----")
    # resume_state = torch.load(torch.load('./{}/pretrained'.format(exp_name)))
    resume_state = torch.load("./ckpt/pretrained")
    net.load_state_dict(resume_state["state_dict"])
    print("----- pre-trained model loaded -----")
else:
    print('--- no weight loaded ---')


# pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# print("Total_params: {}".format(pytorch_total_params))
loss_network = LossNetwork(vgg_model)
loss_network.eval()

# --- Load training data and validation/test data --- #
lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir, rain_L_dir, rain_H_dir, gt_dir), batch_size=train_batch_size, shuffle=True, num_workers=4)

## Uncomment the other validation data loader to keep an eye on performance 
## but note that validating while training significantly increases the train time 

val_data_loader1 = DataLoader(ValData(validate_data_dir, rain_L_dir, rain_H_dir, gt_dir), batch_size=val_batch_size, shuffle=False, num_workers=4)

print("Number of training data: {}".format(len(lbl_train_data_loader)))
print("Number of validation data: {}".format(len(val_data_loader1)))

# --- Previous PSNR and SSIM in testing --- #
net.eval()

################ Note########################

## Uncomment the other validation data loader to keep an eye on performance 
## but note that validating while training significantly increases the test time 

# old_val_psnr, old_val_ssim = validation(net, val_data_loader, device, exp_name)
old_val_psnr1, old_val_ssim1 = validation(net, val_data_loader1, device, exp_name)
# old_val_psnr2, old_val_ssim2 = validation(net, val_data_loader2, device, exp_name)

# print('Rain 800 old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))
print('Rain Drop old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr1, old_val_ssim1))
# print('Test1 old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr2, old_val_ssim2))

net.train()

log_dir = os.path.join("./logs", tensorboard_logdir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
writer = SummaryWriter(log_dir)

count = epoch_start * len(lbl_train_data_loader)

### NOTE: initialization, testing parameters ###
psnr_loss = PSNRLoss(toY=False)
ssim_loss = pytorch_ssim.SSIM()

for epoch in range(epoch_start,num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch)
    for batch_id, train_data in enumerate(lbl_train_data_loader):

        input_image, gt = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        pred_image = net(input_image)

        ### NOTE: trying different loss functions ###
        # smooth_loss = F.smooth_l1_loss(pred_image, gt)
        # smooth_loss = F.mse_loss(pred_image, gt)

        smooth_loss = psnr_loss(pred_image, gt)
        # smooth_loss = psnr_loss(pred_image, gt) - ssim_loss(pred_image, gt)
        # smooth_loss = F.smooth_l1_loss(pred_image, gt) + psnr_loss(pred_image, gt) - ssim_loss(pred_image, gt)

        # smooth_loss = F.cross_entropy(pred_image, gt)
        # smooth_loss = F.kl_div(pred_image, gt) # Kullback-Leibler divergence 
        # smooth_loss = F.nll_loss(pred_image, gt) # negative log likelihood


        perceptual_loss = loss_network(pred_image, gt)
        loss = smooth_loss + lambda_loss*perceptual_loss 

        # loss = F.smooth_l1_loss(pred_image, gt) + smooth_loss


        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))

        count += 1
        writer.add_scalar("Loss/train", loss.item(), count)

        if not (batch_id % 100):
            save_state = {
                            "epoch": epoch,
                            "state_dict": net.state_dict(),
                            "optimizer": optimizer.state_dict()
                        }
            torch.save(save_state, './{}/latest'.format(exp_name))
            print('Epoch: {0}, Iteration: {1}, loss: {2}'.format(epoch, batch_id, loss.item()))

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network parameters --- #
    save_state = {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
    torch.save(save_state, './{}/latest'.format(exp_name))

    # --- Use the evaluation model in testing --- #
    net.eval()

    # val_psnr, val_ssim = validation(net, val_data_loader, device, exp_name)
    # val_psnr1, val_ssim1 = validation(net, val_data_loader1, device, exp_name, psnr_loss, ssim_loss, loss_network)
    # val_psnr2, val_ssim2 = validation(net, val_data_loader2, device, exp_name)

    ######################## NOTE: validation specific steps ###################
    psnr_list = []
    ssim_list = []
    val_lossval_list = []

    for batch_id, val_data in enumerate(val_data_loader1):

        with torch.no_grad():
            input_im, gt, imgid = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            pred_image = net(input_im)
            # compute validation loss
            val_smooth_loss = F.smooth_l1_loss(pred_image, gt) + psnr_loss(pred_image, gt) - ssim_loss(pred_image, gt)
            val_perceptual_loss = loss_network(pred_image, gt)
            val_loss = val_smooth_loss + lambda_loss * val_perceptual_loss 
            val_lossval_list.append(val_loss.item())

        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(pred_image, gt))

    val_psnr1 = sum(psnr_list) / (len(psnr_list) + 1e-10)
    val_ssim1 = sum(ssim_list) / (len(ssim_list) + 1e-10)
    val_lossval = sum(val_lossval_list) / (len(val_lossval_list) + 1e-10)
    ############################################################################

    writer.add_scalar("Validation/PSNR", val_psnr1, count)
    writer.add_scalar("Validation/SSIM", val_ssim1, count)
    writer.add_scalar("Validation/loss", val_lossval, count)

    one_epoch_time = time.time() - start_time
    # print("Rain 800")
    # print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, exp_name)
    print("Rain Drop")
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr1, val_ssim1, exp_name)
    # print("Test1")
    # print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr2, val_ssim2, exp_name)

    # --- update the network weight --- #
    if val_psnr1 >= old_val_psnr1:
        save_state = {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }
        torch.save(save_state, './{}/best'.format(exp_name))
        print('model saved')
        old_val_psnr1 = val_psnr1

        # Note that we find the best model based on validating with raindrop data. 
