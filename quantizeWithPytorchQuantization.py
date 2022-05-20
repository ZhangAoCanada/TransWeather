import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from tabnanny import verbose

import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from train_data_functions import TrainData
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
import torchvision
from torchvision import transforms
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tesnor_qant import QuantDescriptor

from absl import logging
logging.set_verbosity(logging.FATAL)

quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

from pytorch_quantization import quant_modules
quant_modules.initialize()



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

### NOTE: create data loader ###
train_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220325/train"
rain_L_dir = "rain_L"
rain_H_dir = "rain_H"
gt_dir = "gt"
crop_size = [512, 512]
batch_size = 1
data_loader = DataLoader(TrainData(crop_size, train_data_dir, rain_L_dir, rain_H_dir, gt_dir), batch_size=batch_size, shuffle=True, num_workers=4)



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


def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
            
def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            # print(F"{name:40}: {module}")
    # model.cuda()

with torch.no_grad():
    collect_stats(net, data_loader, num_batches=2)
    compute_amax(net, method='percentile', percentile=99.99)

torch.save(net.state_dict(), "./ckpt/quantized_model.pth")




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

torch.onnx.export(net, input_img, "./ckpt/transweather_quant.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=14)
# torch.onnx.export(net, input_img, "./ckpt/transweather_quant.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=14, enable_onnx_checker=False)
# torch.onnx.export(net, input_img, "./ckpt/transweather_quant.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=13, enable_onnx_checker=False,dynamic_axes={'input': {0, 'batch_size'}, 'output': {0, 'batch_size'}})

print("[FINISHED] onnx model exported")

