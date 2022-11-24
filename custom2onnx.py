import sys
from tabnanny import verbose # sys.path.append("/content/drive/MyDrive/DERAIN/TransWeather")

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
# from transweather_model_extra import Transweather
from transweather_model_distillation_customized import TransweatherStudent

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from random import randrange
import torchvision.utils as utils
import cv2
import re
from tqdm import tqdm
from skimage import img_as_ubyte

from torchinfo import summary


def preprocessImage(input_img):
    # Resizing image in the multiple of 16"
    wd_new, ht_new, _ = input_img.shape
    if ht_new>wd_new and ht_new>2048:
        wd_new = int(np.ceil(wd_new*2048/ht_new))
        ht_new = 2048
    elif ht_new<=wd_new and wd_new>2048:
        ht_new = int(np.ceil(ht_new*2048/wd_new))
        wd_new = 2048
    wd_new = int(16*np.ceil(wd_new/16.0))
    ht_new = int(16*np.ceil(ht_new/16.0))
    # input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)
    # input_img = cv2.resize(input_img, (wd_new, ht_new), interpolation=cv2.INTER_AREA)
    input_img = cv2.resize(input_img, (ht_new, wd_new), interpolation=cv2.INTER_AREA)

    # --- Transform to tensor --- #
    # transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # input_im = transform_input(input_img)

    # input_img = input_img / 255.0
    # input_img = (input_img - 0.5) / 0.5
    # transform_input = Compose([ToTensor()])
    # input_im = transform_input(input_img)

    input_im = torch.from_numpy(input_img.astype(np.float32))
    return input_im




val_batch_size = 1
exp_name = "ckpt"
#set seed
seed = 19
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

video_path = "/mnt/d/DATASET/data_capture/CaptureFiles/h97camera_videos/2022_4_12_14_3_53__video.avi"
# output_video_path = "./videos/sample.avi"
model_path = "ckpt/best_StudentModel6.1"

# video = cv2.VideoCapture(video_path)
# video_saving = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc('M','J','P','G'),30,(2040,720))

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = TransweatherStudent()
net = nn.DataParallel(net)

if device == torch.device("cpu"):
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["state_dict"])
    print("====> model ", model_path, " loaded")
else:
    net.load_state_dict(torch.load(model_path)["state_dict"])
    net.to(device)
    print("====> model ", model_path, " loaded")

net.eval()

net = net.module

# sample_img = None
# while True:
#     ret, frame = video.read()
#     if not ret:
#         break
#     sample_image = frame
#     # sample_image = cv2.resize(frame, (960, 540))
#     sample_image = cv2.resize(frame, (640, 368))
#     break

sample_image = cv2.imread("/home/zhangao/DATASET/DATA_20220531/validate/rain_L/Rain_L_2022_5_7_17_16_46_qin_bang_feiting_29.png")
sample_image = cv2.resize(sample_image, (640, 368))

if sample_image is not None:
    print("[INFO] image shape: ", sample_image.shape)
else:
    print("[INFO] image is None")


input_img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
input_img = preprocessImage(input_img)
input_img = input_img.unsqueeze(0)
# input_img = input_img.to(device)

torch.onnx.export(net, input_img, "./ckpt/transweather.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=11)
# torch.onnx.export(net, input_img, "./ckpt/transweather.onnx", verbose=True, export_params=True, do_constant_folding=True, input_names=['input'], output_names=['output'], opset_version=13)

print("[FINISHED] onnx model exported")

