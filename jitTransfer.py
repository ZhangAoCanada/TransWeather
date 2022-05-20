import sys
from tabnanny import verbose
# sys.path.append("/content/drive/MyDrive/DERAIN/TransWeather")

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
from skimage import img_as_ubyte

from torchinfo import summary


def preprocessImage(input_img):
    # Resizing image in the multiple of 16"
    wd_new, ht_new, _ = input_img.shape
    if ht_new>wd_new and ht_new>1024:
        wd_new = int(np.ceil(wd_new*1024/ht_new))
        ht_new = 1024
    elif ht_new<=wd_new and wd_new>1024:
        ht_new = int(np.ceil(ht_new*1024/wd_new))
        wd_new = 1024
    wd_new = int(16*np.ceil(wd_new/16.0))
    ht_new = int(16*np.ceil(ht_new/16.0))
    # input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)
    input_img = cv2.resize(input_img, (wd_new, ht_new), interpolation=cv2.INTER_AREA)

    # --- Transform to tensor --- #
    transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    input_im = transform_input(input_img)

    # input_img = input_img / 255.0
    # input_img = (input_img - 0.5) / 0.5
    # transform_input = Compose([ToTensor()])
    # input_im = transform_input(input_img)
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

video_path = "/home/ao/tmp/clip_videos/h97cam_water_video.mp4"
output_video_path = "./videos/h97cam_water_lambda00_video.avi"
model_path = "ckpt/best_psnr+lambda0.01"

video = cv2.VideoCapture(video_path)
# video_saving = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc('M','J','P','G'),30,(2040,720))

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
    # sample_image = frame
    sample_image = cv2.resize(frame, (640, 480))
    break

if sample_image is not None:
    print("[INFO] image shape: ", sample_image.shape)
else:
    print("[INFO] image is None")


input_img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
input_img = preprocessImage(input_img)
input_img = input_img.unsqueeze(0)
# input_img = input_img.to(device)

torch.onnx.export(net, input_img, "./ckpt/transweather.onnx", verbose=True, input_names=['input'], output_names=['output'])

print("[FINISHED] onnx model exported")




# ### NOTE: start evaluation ###
# with torch.no_grad():
#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break
#         frame = frame[:, 180:1200, :]
#         # pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         input_img = preprocessImage(input_img)
#         input_img = input_img.to(device)
#         input_img = input_img.unsqueeze(0)
#         print("[INFO] ", input_img.shape)
#         pred_image = net(input_img)
#         pred_image_cpu = pred_image[0].permute(1,2,0).cpu().numpy()
#         pred_image_cpu = img_as_ubyte(pred_image_cpu)
#         pred_image_cpu = cv2.resize(pred_image_cpu, (frame.shape[1],frame.shape[0]))
#         image = np.concatenate((frame, pred_image_cpu[..., ::-1]), axis=1)
#         # video_saving.write(image)
#         cv2.imshow("image", image)
#         if cv2.waitKey(1) == 27:
#             break


