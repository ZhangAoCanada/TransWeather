import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import os, glob, re

# --- Validation/test dataset --- #
class TestData(data.Dataset):
    def __init__(self, val_data_dir, rain_L_dir, rain_H_dir, gt_dir, mode="rain_L"):
        super().__init__()
        self.input_names_L, self.gt_names_L = self.getRainLImageNames(val_data_dir, rain_L_dir, gt_dir)
        self.input_names_H, self.gt_names_H = self.getRainHImageNames(val_data_dir, rain_H_dir, gt_dir)
        if mode == "rain_L":
            self.input_names = self.input_names_L
            self.gt_names = self.gt_names_L
        elif mode == "rain_H":
            self.input_names = self.input_names_H
            self.gt_names = self.gt_names_H
        else:
            raise ValueError("Mode must be either 'rain_L' or 'rain_H'")
        self.val_data_dir = val_data_dir

    def getRainLImageNames(self, root_dir, image_dir, gt_dir):
        input_dir = os.path.join(root_dir, image_dir)
        output_dir = os.path.join(root_dir, gt_dir)
        image_names_tmp = []
        image_names = []
        gt_names = []
        for file in os.listdir(input_dir):
            if file.endswith(".png"):
                in_name = os.path.join(input_dir, file)
                image_names_tmp.append(in_name)
        for in_name in image_names_tmp:
            ### NOTE: choice 1 ###
            # image_ind = re.findall(r'\d+', in_name)[0]
            # gt_name = os.path.join(output_dir, image_ind + "_clean.png")
            ### NOTE: choice 2 ###
            gt_name = in_name.replace(image_dir, gt_dir).replace("Rain_L_", "No_Rain_")
            if os.path.exists(gt_name):
                image_names.append(in_name)
                gt_names.append(gt_name)
        return image_names, gt_names

    def getRainHImageNames(self, root_dir, image_dir, gt_dir):
        input_dir = os.path.join(root_dir, image_dir)
        output_dir = os.path.join(root_dir, gt_dir)
        image_names_tmp = []
        image_names = []
        gt_names = []
        for file in os.listdir(input_dir):
            if file.endswith(".png"):
                in_name = os.path.join(input_dir, file)
                image_names_tmp.append(in_name)
        for in_name in image_names_tmp:
            ### NOTE: choice 1 ###
            # image_ind = re.findall(r'\d+', in_name)[0]
            # gt_name = os.path.join(output_dir, image_ind + "_clean.png")
            ### NOTE: choice 2 ###
            gt_name = in_name.replace(image_dir, gt_dir).replace("Rain_H_", "No_Rain_")
            if os.path.exists(gt_name):
                image_names.append(in_name)
                gt_names.append(gt_name)
        return image_names, gt_names

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        input_img = Image.open(input_name)
        gt_img = Image.open(gt_name)

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
        gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)

        return input_im, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
