import sys
sys.path.append("/content/drive/MyDrive/DERAIN/TransWeather")

import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from test_data_functions import TestData
from utils import validation, validation_val, calc_psnr, calc_ssim
import os
import numpy as np
import random
from transweather_model import Transweather

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from random import randrange
import torchvision.utils as utils
import cv2
import re
from tqdm import tqdm

from torchinfo import summary


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default="ckpt", type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name

# test_data_dir = "/content/drive/MyDrive/DERAIN/test"
# image_dir = "data"
# gt_dir = "gt"

test_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220325/test"
rain_L_dir = "rain_L"
rain_H_dir = "rain_H"
gt_dir = "gt"

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Transweather()
summary(net, (1, 3, 720, 480))

net = nn.DataParallel(net, device_ids=device_ids)

if device == torch.device("cpu"):
    net.load_state_dict(torch.load("ckpt/latest", map_location=torch.device('cpu')))
else:
    net.load_state_dict(torch.load("ckpt/best"))
    net.to(device)

net.eval()

val_rain_L_dataset = TestData(test_data_dir, rain_L_dir, rain_H_dir, gt_dir, mode="rain_L")
val_rain_L_data_loader = DataLoader(val_rain_L_dataset, batch_size=1, shuffle=False, num_workers=4)

val_rain_H_dataset = TestData(test_data_dir, rain_L_dir, rain_H_dir, gt_dir, mode="rain_H")
val_rain_H_data_loader = DataLoader(val_rain_H_dataset, batch_size=1, shuffle=False, num_workers=4)

### NOTE: start evaluation ###
bar = tqdm(total = len(val_rain_L_data_loader) + len(val_rain_H_data_loader))
psnr_list = []
ssim_list = []
inference_time_durations = []

for i, data in tqdm(enumerate(val_rain_L_data_loader)):
    bar.update(1)
    with torch.no_grad():
        input_img, gt, imgid = data
        input_img = input_img.to(device)
        gt = gt.to(device)
        start_time = time.time()
        pred_image = net(input_img)
        time_duration = time.time() - start_time
        inference_time_durations.append(time_duration)
        ind = imgid[0].split("/")[-1].split(".")[0]
        ind = re.findall(r'\d+', ind)[0]

        pred_image_images = torch.split(pred_image, 1, dim=0)
        utils.save_image(pred_image_images[0], 'imgs/{}_rainL.png'.format(ind))

        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(pred_image, gt))
        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(pred_image, gt))

avr_psnr = sum(psnr_list) / (len(psnr_list) + 1e-10)
avr_ssim = sum(ssim_list) / (len(ssim_list) + 1e-10)

print("[RainL RESULTS] PSNR: {:.4f}, SSIM: {:.4f}, Average time: {:.4f} ms".format(avr_psnr, avr_ssim, np.mean(inference_time_durations)*1000))

psnr_list_rainH = []
ssim_list_rainH = []
inference_time_durations_rainH = []
for i, data in tqdm(enumerate(val_rain_H_data_loader)):
    bar.update(1)
    with torch.no_grad():
        input_img, gt, imgid = data
        input_img = input_img.to(device)
        gt = gt.to(device)
        start_time = time.time()
        pred_image = net(input_img)
        time_duration = time.time() - start_time
        inference_time_durations.append(time_duration)
        inference_time_durations_rainH.append(time_duration)
        ind = imgid[0].split("/")[-1].split(".")[0]
        ind = re.findall(r'\d+', ind)[0]

        pred_image_images = torch.split(pred_image, 1, dim=0)
        utils.save_image(pred_image_images[0], 'imgs/{}_rainH.png'.format(ind))

        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(pred_image, gt))
        psnr_list_rainH.extend(calc_psnr(pred_image, gt))
        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(pred_image, gt))
        psnr_list_rainH.extend(calc_ssim(pred_image, gt))

avr_psnr_H = sum(psnr_list_rainH) / (len(psnr_list_rainH) + 1e-10)
avr_ssim_H = sum(ssim_list_rainH) / (len(ssim_list_rainH) + 1e-10)

print("[RainH RESULTS] PSNR: {:.4f}, SSIM: {:.4f}, Average time: {:.4f} ms".format(avr_psnr_H, avr_ssim_H, np.mean(inference_time_durations_rainH)*1000))

avr_psnr = sum(psnr_list) / (len(psnr_list) + 1e-10)
avr_ssim = sum(ssim_list) / (len(ssim_list) + 1e-10)

print("[OVERALL RESULTS] PSNR: {:.4f}, SSIM: {:.4f}, Average time: {:.4f} ms".format(avr_psnr, avr_ssim, np.mean(inference_time_durations)*1000))


# ### NOTE: forward 1 image ###
# test_img_name = "/mnt/d/DATASET/Derain_DATA/test/data/0_rain.jpg"
# test_img =  Image.open(test_img_name)

# width, height = test_img.size

# if width < 256 and height < 256 :
#     test_img = test_img.resize((256,256), Image.ANTIALIAS)
# elif width < 256 :
#     test_img = test_img.resize((256,height), Image.ANTIALIAS)
# elif height < 256 :
#     test_img = test_img.resize((width,256), Image.ANTIALIAS)

# width, height = test_img.size

# input_crop_img = test_img

# # --- Transform to tensor --- #
# transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# input_img = transform_input(input_crop_img)

# input_img  = input_img.unsqueeze(0)

# with torch.no_grad():
#     pred_img = net(input_img)
#     pred_image_images = torch.split(pred_img, 1, dim=0)
#     utils.save_image(pred_image_images[0], '/home/ao/image_derain/TransWeather/imgs/test.png')

#     # input_crop_img.show()
