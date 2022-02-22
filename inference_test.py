import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data_functions import ValData
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


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default="ckpt", type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name

# train_data_dir = "/content/drive/MyDrive/DERAIN/train"
# test_data_dir = "/content/drive/MyDrive/DERAIN/test"
train_data_dir = "/mnt/d/DATASET/Derain_DATA/train"
test_data_dir = "/mnt/d/DATASET/Derain_DATA/test"
image_dir = "data"
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


net = nn.DataParallel(net, device_ids=device_ids)

if device == torch.device("cpu"):
    net.load_state_dict(torch.load("/home/ao/image_derain/TransWeather/ckpt/latest", map_location=torch.device('cpu')))
else:
    net.load_state_dict(torch.load("/home/ao/image_derain/TransWeather/ckpt/best"))
    net.to(device)

net.eval()

val_dataset = ValData(test_data_dir, image_dir, gt_dir)
val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)


### NOTE: start evaluation ###
bar = tqdm(total = len(val_dataset))
psnr_list = []
ssim_list = []
inference_time_durations = []
for i, data in tqdm(enumerate(val_data_loader)):
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
        utils.save_image(pred_image_images[0], 'imgs/{}_test.png'.format(ind))

        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(pred_image, gt))
        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(pred_image, gt))

    avr_psnr = sum(psnr_list) / (len(psnr_list) + 1e-10)
    avr_ssim = sum(ssim_list) / (len(ssim_list) + 1e-10)

print("[RESULTS] PSNR: {:.4f}, SSIM: {:.4f}, Average time: {:.4f}".format(avr_psnr, avr_ssim, np.mean(inference_time_durations)))


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
