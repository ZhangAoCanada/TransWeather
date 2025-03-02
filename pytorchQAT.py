import time
import torch
import cv2
import torchvision
import torchvision.transforms as transforms
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
import os, sys
import numpy as np
import random 
from torch.utils.tensorboard import SummaryWriter

from transweather_model_quant import Transweather
from train_psnrloss import PSNRLoss

import torch.quantization
import warnings

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

from torch.quantization import QuantStub, DeQuantStub


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[512, 512], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=10, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
# parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.01, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default="ckpt", type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=1000, type=int)

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs


#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # If GPU is available, change to GPU
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, lambda_loss))


# ##################### NOTE: Change the path to the dataset #####################
# train_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220325/train"
# validate_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220325/validate"
# test_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220325/test"
# rain_L_dir = "rain_L"
# rain_H_dir = "rain_H"
# gt_dir = "gt"

# OverlapPatchEmbed Mlp Attention (Block) Attention_dec (Block_dec) DWConv EncoderTransformer DecoderTransformer convprojection ConvLayer

##################### NOTE: Change the path to the dataset #####################
train_data_dir = "/mnt/d/DATASET/DATA_2070/train"
validate_data_dir = "/mnt/d/DATASET/DATA_2070/validate"
# test_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220325/test"
rain_L_dir = "rain_L"
rain_H_dir = "rain_H"
gt_dir = "gt"


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = Transweather()


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False

# # --- Load the network weight --- #
# if os.path.exists('./{}/'.format(exp_name))==False:     
#     os.mkdir('./{}/'.format(exp_name))  
# try:
#     if os.path.exists("./{}/latest".format(exp_name)):
#         net.load_state_dict(torch.load('./{}/latest'.format(exp_name)))
#         print("----- last trained loaded -----")
#     else:
#         net.load_state_dict(torch.load('./{}/origin_pretrained'.format(exp_name)))
#         print("----- original trained loaded -----")
# except:
#     print('--- no weight loaded ---')

model_path = "./ckpt/best_psnr+lambda0.01"
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
net = net.module


loss_network = LossNetwork(vgg_model)
loss_network.eval()

# --- Load training data and validation/test data --- #
lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir, rain_L_dir, rain_H_dir, gt_dir), batch_size=train_batch_size, shuffle=True, num_workers=4)

## Uncomment the other validation data loader to keep an eye on performance 
## but note that validating while training significantly increases the train time 

val_data_loader1 = DataLoader(ValData(validate_data_dir, rain_L_dir, rain_H_dir, gt_dir), batch_size=val_batch_size, shuffle=False, num_workers=4)

print("Number of training data: {}".format(len(lbl_train_data_loader)))
print("Number of validation data: {}".format(len(val_data_loader1)))

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

video_path = "/home/ao/tmp/clip_videos/h97cam_water_video.mp4"
video = cv2.VideoCapture(video_path)

sample_img = None
while True:
    ret, frame = video.read()
    if not ret:
        break
    sample_image = frame
    # sample_image = cv2.resize(frame, (960, 540))
    sample_image = cv2.resize(frame, (640, 360))
    break

if sample_image is not None:
    print("[INFO] image shape: ", sample_image.shape)
else:
    print("[INFO] image is None")

input_img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
input_img = preprocessImage(input_img)
input_img = input_img.unsqueeze(0)
# input_img = input_img.to(device)



net.eval()
# net.train()
net.qconfig = torch.quantization.default_qconfig
# net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
net.fuse_model()

torch.quantization.prepare(net, inplace=True)
# torch.quantization.prepare_qat(net, inplace=True)

net(input_img)

net_quant = torch.quantization.convert(net, inplace=False)

print("[FINISHED] model conversion is finished.")

torch.onnx.export(net_quant, input_img, "./ckpt/transweather_pytorchPostTrainingQuant.onnx", verbose=True, input_names=['input'], output_names=['output'], opset_version=13)
# # torch.onnx.export(net, input_img, "./ckpt/transweather_pytorchPostTrainingQuant.onnx", verbose=True, export_params=True, do_constant_folding=True, input_names=['input'], output_names=['output'], opset_version=11)

print("[FINISHED] onnx model exported")



while True:
    ret, frame = video.read()
    if not ret:
        break
    sample_image = frame
    # sample_image = cv2.resize(frame, (960, 540))
    sample_image = cv2.resize(frame, (640, 360))
    sample_image = preprocessImage(sample_image).unsqueeze(0)
    print("---", sample_image.shape)
    pred = net_quant(sample_image)
    # pred = pred[0].detach().cpu().numpy()
    pred = pred[0].numpy()
    pred = pred * 255.0
    pred = pred.astype(np.uint8)
    pred = cv2.resize(pred, (640, 360))
    cv2.imshow("pred", pred)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # break




# net.train()

# log_dir = "./logs/2070_images"
# if not os.path.exists(log_dir):
#     os.mkdir(log_dir)
# writer = SummaryWriter(log_dir)

# count = 0

# ### NOTE: initialization, testing parameters ###
# psnr_loss = PSNRLoss(toY=False)

# for epoch in range(epoch_start,num_epochs):
#     psnr_list = []
#     start_time = time.time()
#     adjust_learning_rate(optimizer, epoch)
#     for batch_id, train_data in enumerate(lbl_train_data_loader):

#         input_image, gt = train_data
#         input_image = input_image.to(device)
#         gt = gt.to(device)

#         # --- Zero the parameter gradients --- #
#         optimizer.zero_grad()

#         # --- Forward + Backward + Optimize --- #
#         net.train()
#         pred_image = net(input_image)

#         ### NOTE: trying different loss functions ###
#         smooth_loss = psnr_loss(pred_image, gt)
        
#         perceptual_loss = loss_network(pred_image, gt)
#         loss = smooth_loss + lambda_loss*perceptual_loss 

#         loss.backward()
#         optimizer.step()

#         # --- To calculate average PSNR --- #
#         psnr_list.extend(to_psnr(pred_image, gt))

#         count += 1
#         writer.add_scalar("Loss/train", loss.item(), count)

#         if not (batch_id % 100):
#             torch.save(net.state_dict(), './{}/latest'.format(exp_name))
#             print('Epoch: {0}, Iteration: {1}, loss: {2}'.format(epoch, batch_id, loss.item()))

#     # --- Calculate the average training PSNR in one epoch --- #
#     train_psnr = sum(psnr_list) / len(psnr_list)

#     # --- Save the network parameters --- #
#     torch.save(net.state_dict(), './{}/latest'.format(exp_name))

#     # --- Use the evaluation model in testing --- #
#     net.eval()

#     # val_psnr, val_ssim = validation(net, val_data_loader, device, exp_name)
#     val_psnr1, val_ssim1 = validation(net, val_data_loader1, device, exp_name)
#     # val_psnr2, val_ssim2 = validation(net, val_data_loader2, device, exp_name)
#     writer.add_scalar("Validation/PSNR", val_psnr1, count)
#     writer.add_scalar("Validation/SSIM", val_ssim1, count)

#     one_epoch_time = time.time() - start_time
#     print("Rain Drop")
#     print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr1, val_ssim1, exp_name)

#     # --- update the network weight --- #
#     if val_psnr1 >= old_val_psnr1:
#         torch.save(net.state_dict(), './{}/best'.format(exp_name))
#         print('model saved')
#         old_val_psnr1 = val_psnr1
