import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# from train_data_functions_seq import TrainData
from train_data_functions_seq_batch import TrainData
from test_data_functions_seq import TestData as ValData
from utils import to_psnr, print_log, validation_seq, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
import os
import numpy as np
import random 
from torch.utils.tensorboard import SummaryWriter

from transweather_sequence import TransweatherSeq
# from transweather_model import Transweather
from transweather_model_teacher import TransweatherTeacher

from train_psnrloss import PSNRLoss
import pytorch_ssim

from utils import calc_psnr, calc_ssim

import cv2


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[512, 512], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=1, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.01, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default="ckpt", type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=1000, type=int)
parser.add_argument('-logdir', help='for tensorboard', default="seq3", type=str)

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
train_data_dir = "/home/za/DATASET/SPAC/SPAC/Dataset_Training_Synthetic"
validate_data_dir = "/home/za/DATASET/SPAC/SPAC/Dataset_Testing_Synthetic"
test_data_dir = "/home/za/DATASET/SPAC/SPAC/Dataset_Testing_RealRain"
train_gt_dir = ["GT"]
train_rain_dir = ["Rain_01", "Rain_02", "Rain_03"]
train_sequence = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
validate_gt_dir = ["GT"]
validate_rain_dir = ["Rain"]
validate_sequence = ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"]

# transweather_pretrained_pth = "./ckpt/transweather_pretrained.pth"
transweather_pretrained_pth = "./ckpt/normal_pretrained.pth"
ckpt_path = "./ckpt/whatever"

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = TransweatherSeq(transweather_pretrained_pth)
# net = TransweatherTeacher(transweather_pretrained_pth)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

# modelbest_path = "./{}/best_seq".format(exp_name)

# if os.path.exists('./{}/'.format(exp_name))==False:     
#     os.mkdir('./{}/'.format(exp_name))  

# if os.path.exists(modelbest_path):
#     resume_state = torch.load(modelbest_path)
#     net.load_state_dict(resume_state["state_dict"])
#     optimizer.load_state_dict(resume_state["optimizer"])
#     epoch_start = resume_state["epoch"]
#     print("----- model '{}' loaded -----".format(modelbest_path))
# elif os.path.exists(ckpt_path):
#     resume_state = torch.load(ckpt_path)
#     net.load_state_dict(resume_state["state_dict"])
#     print("----- model '{}' loaded -----".format(ckpt_path))
# else:
#     print('--- no weight loaded ---')


# loss_network = LossNetwork(vgg_model)
# loss_network.eval()

lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir, train_gt_dir, train_rain_dir, train_sequence), batch_size=train_batch_size, shuffle=False, num_workers=1)

# lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir, train_gt_dir, train_rain_dir, train_sequence), batch_size=8, shuffle=True, num_workers=4)

val_data_loader1 = DataLoader(ValData(validate_data_dir, validate_gt_dir, validate_rain_dir, validate_sequence), batch_size=val_batch_size, shuffle=False, num_workers=1)

print("[INFO] Number of training data: {}".format(len(lbl_train_data_loader)))
print("[INFO] Number of validation data: {}".format(len(val_data_loader1)))

# net.eval()

# old_val_psnr1, old_val_ssim1 = validation_seq(net, val_data_loader1, device, exp_name)

# print('Rain Drop old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr1, old_val_ssim1))

net.train()

log_dir = os.path.join("./logs_seq", tensorboard_logdir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
writer = SummaryWriter(log_dir)

count = epoch_start * len(lbl_train_data_loader)

psnr_loss = PSNRLoss(toY=False)
ssim_loss = pytorch_ssim.SSIM()

for epoch in range(epoch_start,num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch)
    epoch_reset = True
    for batch_id, train_data  in enumerate(lbl_train_data_loader):

        input_image, gt, is_continue = train_data

        input_image = torch.concat(input_image, dim=0)
        gt = torch.concat(gt, dim=0)
        is_continue = is_continue[0]

        input_image = input_image.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()
        net.train()
        if_reset = epoch_reset if epoch_reset else not is_continue
        pred_image = net(input_image, reset=if_reset)
        epoch_reset = False

        ### NOTE: trying different loss functions ###
        # smooth_loss = F.smooth_l1_loss(pred_image, gt)
        # smooth_loss = F.mse_loss(pred_image, gt)
        # smooth_loss = psnr_loss(pred_image, gt) + F.smooth_l1_loss(pred_image, gt)
        # smooth_loss = psnr_loss(pred_image, gt) - ssim_loss(pred_image, gt)
        # smooth_loss = F.smooth_l1_loss(pred_image, gt) + psnr_loss(pred_image, gt) - ssim_loss(pred_image, gt)
        # smooth_loss = F.cross_entropy(pred_image, gt)
        # smooth_loss = F.kl_div(pred_image, gt) # Kullback-Leibler divergence 
        # smooth_loss = F.nll_loss(pred_image, gt) # negative log likelihood

        # perceptual_loss = loss_network(pred_image, gt)
        # loss = smooth_loss + lambda_loss*perceptual_loss 

        loss = psnr_loss(pred_image, gt) + F.smooth_l1_loss(pred_image, gt)


        loss.backward()
        optimizer.step()

        psnr_list.extend(to_psnr(pred_image, gt))

        count += 1
        writer.add_scalar("Loss/train", loss.item(), count)

        if not (batch_id % 100):
            save_state = {
                            "epoch": epoch,
                            "state_dict": net.state_dict(),
                            "optimizer": optimizer.state_dict()
                        }
            torch.save(save_state, './{}/latest_seq'.format(exp_name))
            print('Epoch: {0}, Iteration: {1}, loss: {2}'.format(epoch, batch_id, loss.item()))

    train_psnr = sum(psnr_list) / len(psnr_list)

    save_state = {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
    torch.save(save_state, './{}/latest_seq'.format(exp_name))

    net.eval()

    psnr_list = []
    ssim_list = []
    val_lossval_list = []

    eval_start = True
    for batch_id, val_data in enumerate(val_data_loader1):

        with torch.no_grad():
            input_im, gt, if_continue, imgid = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            if_reset = eval_start if eval_start else not if_continue
            pred_image = net(input_im, reset=if_reset)
            eval_start = False

            # val_smooth_loss = F.smooth_l1_loss(pred_image, gt) + psnr_loss(pred_image, gt) - ssim_loss(pred_image, gt)
            # val_perceptual_loss = loss_network(pred_image, gt)
            # val_loss = val_smooth_loss + lambda_loss * val_perceptual_loss 

            val_loss = F.smooth_l1_loss(pred_image, gt) + psnr_loss(pred_image, gt) - ssim_loss(pred_image, gt)
            val_lossval_list.append(val_loss.item())

        psnr_list.extend(calc_psnr(pred_image, gt))
        ssim_list.extend(calc_ssim(pred_image, gt))

    val_psnr1 = sum(psnr_list) / (len(psnr_list) + 1e-10)
    val_ssim1 = sum(ssim_list) / (len(ssim_list) + 1e-10)
    val_lossval = sum(val_lossval_list) / (len(val_lossval_list) + 1e-10)

    writer.add_scalar("Validation/PSNR", val_psnr1, count)
    writer.add_scalar("Validation/SSIM", val_ssim1, count)
    writer.add_scalar("Validation/loss", val_lossval, count)

    one_epoch_time = time.time() - start_time
    print("Rain Drop")
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr1, val_ssim1, exp_name)

    if val_psnr1 >= old_val_psnr1:
        save_state = {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }
        torch.save(save_state, './{}/best_seq'.format(exp_name))
        print('model saved')
        old_val_psnr1 = val_psnr1
