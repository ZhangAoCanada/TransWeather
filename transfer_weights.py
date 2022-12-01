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
from transweather_model_teacher import TransweatherTeacher

from train_psnrloss import PSNRLoss
import pytorch_ssim

from utils import calc_psnr, calc_ssim




model_load_path = "./ckpt/best_seq_normal"
# model_load_path = "./ckpt/best_combinedData"
model_save_path = "./ckpt/normal_pretrained.pth"




print("************** start model conversion **************")

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
# net = Transweather()
net = TransweatherTeacher()



# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False


resume_state = torch.load(model_load_path)
net.load_state_dict(resume_state["state_dict"])
# net.load_state_dict(resume_state)
print("====> loaded checkpoint '{}'".format(model_load_path))

net.module.save(model_save_path)