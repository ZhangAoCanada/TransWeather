from pkg_resources import invalid_marker
import torch.utils.data as data
from PIL import Image
from random import randrange, shuffle
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
from os import path
import numpy as np
import torch
import glob, os, random
import torchvision.transforms.functional as TF
from tqdm import tqdm
from glob import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, gt_dir, rain_dir, sequences):
        super().__init__()
        assert isinstance(gt_dir, list)
        assert isinstance(rain_dir, list)
        assert isinstance(sequences, list)
        assert len(gt_dir) == 1
        gt_dir = gt_dir[0]
        print("Reading Data...")
        self.gt_names, self.input_names, self.seq_len = self.getAllImageNames(
                        train_data_dir, gt_dir, rain_dir, sequences)
        print("====> [INFO] Total number of training data: ", len(self.gt_names))
        print("====> [INFO] Training sequence points: ", self.seq_len)
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir
        self.seq_index = 0
    
    def getAllImageNames(self, train_data_dir, gt_dir, rain_dir, sequences):
        gt_imgs = []
        rain_imgs = []
        seq_len = []
        seq_count = 0
        for seq_ind in range(len(sequences)):
            seq = sequences[seq_ind]
            train_gt_dir = os.path.join(train_data_dir, seq + "_" + gt_dir)
            gt_img_names = glob(os.path.join(train_gt_dir, "*.jpg"))
            for gt_img_n in gt_img_names:
                rain_sub_img_names = []
                for rain_sub in rain_dir:
                    train_rain_sub_dir = os.path.join(
                                        train_data_dir, seq + "_" + rain_sub)
                    rain_sub_img_names.append(gt_img_n.replace(train_gt_dir, train_rain_sub_dir))
                rain_imgs.append(rain_sub_img_names)
            seq_count += len(gt_img_names)
            seq_len.append(seq_count)
            gt_imgs.extend(gt_img_names)
        assert len(gt_imgs) == len(rain_imgs)
        assert len(gt_imgs) == seq_len[-1]
        return gt_imgs, rain_imgs, seq_len

    def get_images(self, index):
        input_names = self.input_names[index]
        gt_name = self.gt_names[index]

        input_imgs = [Image.open(input_name) for input_name in input_names]

        try:
            gt_img = Image.open(gt_name)
        except:
            gt_img = Image.open(gt_name).convert('RGB')
        
        gt_imgs = [gt_img] * len(input_imgs)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        input_ims = [transform_input(input_img) for input_img in input_imgs]
        gts = [transform_gt(gt_img) for gt_img in gt_imgs]

        # --- Normalize the input image --- #
        normalize_input = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_ims = [normalize_input(input_im) for input_im in input_ims]
        
        is_continue = True
        if index >= self.seq_len[self.seq_index]:
            self.seq_index += 1
            is_continue = False
            self.seq_index = 0 if self.seq_index >= len(self.seq_len) else self.seq_index
        return input_ims, gts, is_continue

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
