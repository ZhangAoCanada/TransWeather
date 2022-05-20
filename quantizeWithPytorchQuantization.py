import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from tabnanny import verbose

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
from transweather_model_extra import Transweather

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from random import randrange
import torchvision.utils as utils
import cv2
import re
from tqdm import tqdm
from skimage import img_as_ubyte

from torchinfo import summary

### NOTE: for quantization ###
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tesnor_qant import QuantDescriptor
# from pytorch_quantization import quant_modules
# quant_nn.TensorQuantizer.use_fb_fake_quant = True
# quant_modules.initialize()


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
    input_img = cv2.resize(input_img, (ht_new, wd_new), interpolation=cv2.INTER_AREA)
    input_im = torch.from_numpy(input_img.astype(np.float32))
    return input_im


video_path = "videos/dusty_video_960_540.avi"
model_path = "ckpt/best_psnr+lambda0.01"

video = cv2.VideoCapture(video_path)

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Transweather()
net = nn.DataParallel(net)

if device == torch.device("cpu"):
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("====> model ", model_path, " loaded")
else:
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    print("====> model ", model_path, " loaded")

net.eval()

net = net.module

sample_img = None
while True:
    ret, frame = video.read()
    if not ret:
        break
    sample_image = frame
    sample_image = cv2.resize(frame, (960, 540))
    # sample_image = cv2.resize(frame, (640, 360))
    break

if sample_image is not None:
    print("[INFO] image shape: ", sample_image.shape)
else:
    print("[INFO] image is None")


input_img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
input_img = preprocessImage(input_img)
input_img = input_img.unsqueeze(0)

torch.onnx.export(net, input_img, "./ckpt/transweather_quant.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=14, enable_onnx_checker=False)
# torch.onnx.export(net, input_img, "./ckpt/transweather_quant.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=13, enable_onnx_checker=False,dynamic_axes={'input': {0, 'batch_size'}, 'output': {0, 'batch_size'}})

print("[FINISHED] onnx model exported")

