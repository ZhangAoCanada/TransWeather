import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data_functions import TrainData
from val_data_functions import ValData
from utils import to_psnr, print_log, validationDistillation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
import os
import numpy as np
import random 
from torch.utils.tensorboard import SummaryWriter

from transweather_model_teacher import TransweatherTeacher
from transweather_model_distillation import TransweatherStudent

from train_psnrloss import PSNRLoss

### NOTE: add GD path ###
# import sys
# sys.path.append('/content/drive/MyDrive/train/data')

### NOTE: backend agg ###
# plt.switch_backend('agg')

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
parser.add_argument('-distillation_scale', help='for model distillation', default=0.9, type=int)
parser.add_argument('-logdir', help='for tensorboard', default="StudentModel6.1", type=str)

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
distillation_scale = args.distillation_scale
tensorboard_logdir = args.logdir

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # If GPU is available, change to GPU
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}\ndistillation scale: {}'.format(learning_rate, crop_size, train_batch_size, val_batch_size, lambda_loss, distillation_scale))


##################### NOTE: Change the path to the dataset #####################
# train_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220617/train"
# validate_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220617/validate"
# test_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220617/test"
# train_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220531/train"
# validate_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220531/validate"
# test_data_dir = "/content/drive/MyDrive/DERAIN/DATA_20220531/test"
train_data_dir = "/home/zhangao/DATASET/DATA_20220617/train"
validate_data_dir = "/home/zhangao/DATASET/DATA_20220617/validate"
test_data_dir = "/home/zhangao/DATASET/DATA_20220617/test_specific"
rain_L_dir = "rain_L"
rain_H_dir = "rain_H"
gt_dir = "gt"
teacher_model_path = './{}/best_combinedData'.format(exp_name)
# teacher_model_path = './{}/best_enhancedData'.format(exp_name)

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
net_teacher = TransweatherTeacher()
net_student = TransweatherStudent()

# --- Build optimizer --- #
optimizer = torch.optim.Adam(net_student.parameters(), lr=learning_rate)

# --- Multi-GPU --- #
net_teacher = net_teacher.to(device)
net_student = net_student.to(device)
net_teacher = nn.DataParallel(net_teacher, device_ids=device_ids)
net_student = nn.DataParallel(net_student, device_ids=device_ids)

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False

# --- Load the network weight --- #
if os.path.exists('./{}/'.format(exp_name))==False:     
    os.mkdir('./{}/'.format(exp_name))  
try:
    net_teacher.load_state_dict(torch.load(teacher_model_path))
    print("----- teacher model {} loaded -----".format(teacher_model_path))
except:
    print('--- teacher model no weight loaded ---')

student_model_path = './{}/best'.format(exp_name)
try:
    if os.path.exists("./{}/best".format(exp_name)):
        resume_state = torch.load('./{}/best'.format(exp_name))
        net_student.load_state_dict(resume_state["state_dict"])
        optimizer.load_state_dict(resume_state["optimizer"])
        epoch_start = resume_state["epoch"]
        print("----- student model best trained loaded -----")
    else:
        resume_state = torch.load(student_model_path)
        net_student.load_state_dict(resume_state["state_dict"])
        optimizer.load_state_dict(resume_state["optimizer"])
        epoch_start = resume_state["epoch"]
        print("----- student model {} loaded -----".format(student_model_path))
except:
    print('--- student model no weight loaded ---')

# student_model_path = './{}/best'.format(exp_name)
# try:
#     if os.path.exists("./{}/best".format(exp_name)):
#         resume_state = torch.load('./{}/best'.format(exp_name))
#         net_student.load_state_dict(resume_state)
#         print("----- student model best trained loaded -----")
#     else:
#         resume_state = torch.load(student_model_path)
#         net_student.load_state_dict(resume_state)
#         print("----- student model {} loaded -----".format(student_model_path))
# except:
#     print('--- student model no weight loaded ---')

loss_network = LossNetwork(vgg_model)
loss_network.eval()

# --- Load training data and validation/test data --- #
lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir, rain_L_dir, rain_H_dir, gt_dir), batch_size=train_batch_size, shuffle=True, num_workers=4)

## Uncomment the other validation data loader to keep an eye on performance 
## but note that validating while training significantly increases the train time 

val_data_loader1 = DataLoader(ValData(validate_data_dir, rain_L_dir, rain_H_dir, gt_dir), batch_size=val_batch_size, shuffle=False, num_workers=4)

print("Number of training data: {}".format(len(lbl_train_data_loader)))
print("Number of validation data: {}".format(len(val_data_loader1)))

# --- Previous PSNR and SSIM in testing --- #
net_teacher.eval()
teacher_val_psnr1, teacher_val_ssim1 = validationDistillation(net_teacher, val_data_loader1, device, exp_name)
print('Teacher model val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(teacher_val_psnr1, teacher_val_ssim1))

student_val_psnr1, student_val_ssim1 = validationDistillation(net_student, val_data_loader1, device, exp_name)
print('Student model val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(student_val_psnr1, student_val_ssim1))
old_val_psnr1 = student_val_psnr1
old_val_ssim1 = student_val_ssim1

net_teacher.eval()
net_student.train()

print("[INFO] Teacher model encoder depths: {}, decoder depths: {}.".format(net_teacher.module.Tenc.depths, net_teacher.module.Tdec.depths[0]))
print("[INFO] Student model encoder depths: {}, decoder depths: {}.".format(net_student.module.Tenc.depths, net_student.module.Tdec.depths[0]))

log_dir = os.path.join("./logs", tensorboard_logdir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
writer = SummaryWriter(log_dir)

count = epoch_start * len(lbl_train_data_loader)

### NOTE: initialization, testing parameters ###
psnr_loss = PSNRLoss(toY=False)

for epoch in range(epoch_start,num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch)
    for batch_id, train_data in enumerate(lbl_train_data_loader):

        input_image, gt = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        pred_list_teacher = net_teacher(input_image)
        pred_list_student = net_student(input_image)

        assert(len(pred_list_student) == len(pred_list_teacher))
        assert(len(pred_list_student) > 0)

        # --- Model distillation --- #
        distillation_loss = 0.0
        for i in range(len(pred_list_student)):
            # distillation_loss += F.smooth_l1_loss(pred_list_student[i], pred_list_teacher[i])
            distillation_loss += psnr_loss(pred_list_student[i], pred_list_teacher[i])

        # --- NOTE: trying different loss functions --- #
        # smooth_loss = F.smooth_l1_loss(pred_image, gt)
        # smooth_loss = F.mse_loss(pred_image, gt)
        smooth_loss = psnr_loss(pred_list_student[-1], gt)
        # smooth_loss = F.cross_entropy(pred_image, gt)
        # smooth_loss = F.kl_div(pred_image, gt) # Kullback-Leibler divergence 
        # smooth_loss = F.nll_loss(pred_image, gt) # negative log likelihood
        
        perceptual_loss = loss_network(pred_list_student[-1], gt)
        pred_loss = smooth_loss + lambda_loss*perceptual_loss 

        # --- combine... --- #
        loss = distillation_scale * distillation_loss + (1-distillation_scale) * pred_loss

        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(pred_list_student[-1], gt))

        count += 1
        writer.add_scalar("Loss/train", loss.item(), count)

        if not (batch_id % 100):
            save_state = {
                            'epoch': epoch,
                            'state_dict': net_student.state_dict(),
                            'optimizer': optimizer.state_dict()
                        }
            torch.save(save_state, './{}/latest'.format(exp_name))
            print('Epoch: {0}, Iteration: {1}, loss: {2}'.format(epoch, batch_id, loss.item()))

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network parameters --- #
    save_state = {
                    'epoch': epoch,
                    'state_dict': net_student.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
    torch.save(save_state, './{}/latest'.format(exp_name))

    # --- Use the evaluation model in testing --- #
    net_student.eval()

    # val_psnr, val_ssim = validation(net, val_data_loader, device, exp_name)
    val_psnr1, val_ssim1 = validationDistillation(net_student, val_data_loader1, device, exp_name)
    # val_psnr2, val_ssim2 = validation(net, val_data_loader2, device, exp_name)
    writer.add_scalar("Validation/PSNR", val_psnr1, count)
    writer.add_scalar("Validation/SSIM", val_ssim1, count)

    one_epoch_time = time.time() - start_time
    print('[INFO] ({0:.0f}s) Epoch [{1}/{2}], Teacher Val_PSNR: {3:.2f}, Teacher Val_SSIM: {4:.2f}, Val_PSNR:{5:.2f}, Val_SSIM:{6:.2f}'.format(one_epoch_time, epoch+1, num_epochs, teacher_val_psnr1, teacher_val_ssim1, val_psnr1, val_ssim1))

    # --- update the network weight --- #
    if val_psnr1 >= old_val_psnr1:
        save_state = {
                        'epoch': epoch,
                        'state_dict': net_student.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
        torch.save(save_state, './{}/best'.format(exp_name))
        print('model saved')
        old_val_psnr1 = val_psnr1
        old_val_ssim1 = val_ssim1
