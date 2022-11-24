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

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
quant_nn.TensorQuantizer.use_fb_fake_quant = True
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
    input_im = torch.from_numpy(input_img.astype(np.float32) / 255.)
    return input_im


video_path = "/home/ao/tmp/clip_videos/h97cam_water_video.mp4"
output_video_path = "./videos/h97cam_water_lambda00_video.avi"
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



### NOTE: dynamic quantization with pytorch natively ###
net_int8 = torch.quantization.quantize_dynamic(net, {torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d}, dtype=torch.qint8)
# net_int8 = torch.quantization.quantize_dynamic(net, dtype=torch.qint8)




sample_img = None
while True:
    ret, frame = video.read()
    if not ret:
        break
    # sample_image = frame
    # sample_image = cv2.resize(frame, (960, 540))
    sample_image = cv2.resize(frame, (640, 360))
    sample_image_show = sample_image.copy()
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    sample_image = preprocessImage(sample_image)
    sample_image = sample_image.unsqueeze(0)
    start = time.time()
    pred = net_int8(sample_image)
    # pred = net(sample_image)
    pred = pred[0].detach().numpy()
    pred = img_as_ubyte(pred)
    pred = cv2.resize(pred, (sample_image_show.shape[1], sample_image_show.shape[0]))
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    image = np.concatenate([sample_image_show, pred], axis=1)
    print("[INFO] inference time: ", time.time() - start)
    # cv2.imshow("img", image)
    # if cv2.waitKey(1) == 27: break
    # break
    break


if sample_image is not None:
    print("[INFO] image shape: ", sample_image.shape)
else:
    print("[INFO] image is None")


# torch.onnx.export(net_int8, sample_image, "./ckpt/transweather_quant.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=11, enable_onnx_checker=False)
torch.onnx.export(net_int8, sample_image, "./ckpt/transweather_quant.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=13)

print("[FINISHED] onnx model exported")

