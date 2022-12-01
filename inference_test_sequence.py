import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from genericpath import exists
import sys
# sys.path.append("/content/drive/MyDrive/DERAIN/TransWeather")

import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from test_data_functions_seq import TestData, InferData
from utils import validation, validation_val, calc_psnr, calc_ssim
import os
import numpy as np
import random
from transweather_sequence import TransweatherSeq
# from transweather_model import Transweather
from transweather_model_teacher import TransweatherTeacher

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from random import randrange
import torchvision.utils as utils
import cv2
import re
from tqdm import tqdm

from torchinfo import summary
import shutil


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default="ckpt", type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name

train_data_dir = "/home/za/DATASET/SPAC/SPAC/Dataset_Training_Synthetic"
validate_data_dir = "/home/za/DATASET/SPAC/SPAC/Dataset_Testing_Synthetic"
test_data_dir = "/home/za/DATASET/SPAC/SPAC/Dataset_Testing_RealRain"
train_gt_dir = ["GT"]
train_rain_dir = ["Rain_01", "Rain_02", "Rain_03"]
train_sequence = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
validate_gt_dir = ["GT"]
validate_rain_dir = ["Rain"]
validate_sequence = ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"]
test_rain_dir = ["Rain"]
test_sequence = ["ra1", "ra2", "ra3", "ra4", "rb1", "rb2", "rb3"]

ckpt_path = "./ckpt/best_seq"
mode = "evaluate" # ["evaluate", "inference"]

### NOTE: clear directory "./imgs" before inference ###
for root, dirs, files in os.walk("./imgs"):
    for file in files:
        os.remove(os.path.join(root, file))
    # remove subdirectories
    for subdir in dirs:
        shutil.rmtree(os.path.join(root, subdir))


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

# net = TransweatherTeacher()
net = TransweatherSeq()
summary(net, (1, 3, 720, 480))

net = nn.DataParallel(net, device_ids=device_ids)

if device == torch.device("cpu"):
    net.load_state_dict(torch.load("ckpt/latest", map_location=torch.device('cpu')))
else:
    resume_state = torch.load(ckpt_path)
    net.load_state_dict(resume_state["state_dict"])
    net.to(device)
    print("====> model {} loaded".format(ckpt_path))

net.eval()

if mode == "evaluate":
    test_data_loader = DataLoader(
        TestData(validate_data_dir, validate_gt_dir, 
                validate_rain_dir, validate_sequence), 
        batch_size=1, shuffle=False, num_workers=1
        )

    psnr_list = []
    ssim_list = []
    inference_time_durations = []
    bar = tqdm(total=len(test_data_loader))
    eval_start = True
    for i, test_data in enumerate(test_data_loader):
        bar.update(1)

        with torch.no_grad():
            input_im, gt, if_continue, imgid = test_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            if_reset = eval_start if eval_start else not if_continue
            start_time = time.time()
            pred_image = net(input_im, reset=if_reset)
            inference_time_durations.append(time.time() - start_time)
            eval_start = False

            ind = imgid[0].split("/")[-1].split(".")[0]
            sub_dir = imgid[0].split("/")[-2]
            if not os.path.exists("./imgs/{}".format(sub_dir)):
                os.makedirs("./imgs/{}".format(sub_dir))
            pred_img_single = torch.split(pred_image, 1, dim=0)
            input_img = torch.split(input_im, 1, dim=0)[0]
            input_img = input_img * 0.5 + 0.5
            img_to_save = torch.cat((input_img, pred_img_single[0], gt), dim=0)
            utils.save_image(img_to_save, "./imgs/{}/{}_pred.png".format(sub_dir, ind))

            psnr_list.extend(calc_psnr(pred_image, gt))
            ssim_list.extend(calc_ssim(pred_image, gt))

    test_psnr = sum(psnr_list) / (len(psnr_list) + 1e-10)
    test_ssim = sum(ssim_list) / (len(ssim_list) + 1e-10)
    test_speed = sum(inference_time_durations) / (len(inference_time_durations) + 1e-10)

    print("------------ RESULTS on Dataset_Testing_Synthetic ------------")
    print("PSNR: {:.4f} SSIM: {:.4f} Speed: {:.4f}".format(test_psnr, test_ssim, test_speed))


elif mode == "inference":
    inferece_data_loader = DataLoader(
        InferData(test_data_dir, test_rain_dir, test_sequence),
        batch_size=1, shuffle=False, num_workers=1
        )
    
    inference_time_durations = []
    bar = tqdm(total=len(inferece_data_loader))
    eval_start = True
    for i, inference_data in enumerate(inferece_data_loader):
        bar.update(1)

        with torch.no_grad():
            input_im, if_continue, imgid = inference_data
            input_im = input_im.to(device)
            if_reset = eval_start if eval_start else not if_continue
            start_time = time.time()
            pred_image = net(input_im, reset=if_reset)
            inference_time_durations.append(time.time() - start_time)
            eval_start = False

            ind = imgid[0].split("/")[-1].split(".")[0]
            sub_dir = imgid[0].split("/")[-2]
            if not os.path.exists("./imgs/{}".format(sub_dir)):
                os.makedirs("./imgs/{}".format(sub_dir))
            pred_img_single = torch.split(pred_image, 1, dim=0)
            input_img = torch.split(input_im, 1, dim=0)[0]
            input_img = input_img * 0.5 + 0.5
            img_to_save = torch.cat((input_img, pred_img_single[0]), dim=0)
            utils.save_image(img_to_save, "./imgs/{}/{}_pred.png".format(sub_dir, ind))

    test_speed = sum(inference_time_durations) / (len(inference_time_durations) + 1e-10)
    print("------------ RESULTS on Dataset_Testing_RealRain ------------")
    print("Speed: {:.4f}".format(test_speed))
    
else:
    raise ValueError("mode should be either 'evaluate' or 'inference'.")