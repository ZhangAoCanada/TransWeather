import sys
sys.path.append("/content/drive/MyDrive/DERAIN/TransWeather")

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
    wd_new,ht_new = input_img.size
    if ht_new>wd_new and ht_new>1024:
        wd_new = int(np.ceil(wd_new*1024/ht_new))
        ht_new = 1024
    elif ht_new<=wd_new and wd_new>1024:
        ht_new = int(np.ceil(ht_new*1024/wd_new))
        wd_new = 1024
    wd_new = int(16*np.ceil(wd_new/16.0))
    ht_new = int(16*np.ceil(ht_new/16.0))
    input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)

    # --- Transform to tensor --- #
    transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    input_im = transform_input(input_img)
    return input_im



# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default="ckpt", type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name

video_path = "/content/drive/MyDrive/DERAIN/DATA_captured/something_else/sample_video.mp4"
output_video_path = "./videos/result_video.mp4"

video = cv2.VideoCapture(video_path)
video_saving = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc('M','J','P','G'),30,(1280,720))

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


### NOTE: start evaluation ###
with torch.no_grad():
    while True:
        ret, frame = video.read()
        if not ret:
            break
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_img = preprocessImage(pil_img)
        input_img = input_img.to(device)
        input_img = input_img.unsqueeze(0)
        pred_image = net(input_img)
        pred_image_cpu = pred_image[0].view(pred_image.shape[2],pred_image.shape[3],3).detach().cpu().numpy()
        pred_image_cpu = img_as_ubyte(pred_image_cpu)
        pred_image_cpu = cv2.resize(pred_image_cpu, (frame.shape[1],frame.shape[0]))
        image = np.concatenate((frame, pred_image_cpu), axis=1)
        video_saving.write(image)


