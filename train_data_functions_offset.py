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

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, rain_L_dir, rain_H_dir, gt_dir):
        super().__init__()
        if rain_H_dir is None:
            self.input_names, self.gt_names = self.getImageNames(train_data_dir, rain_L_dir, gt_dir)
        else:
            self.input_names_L, self.gt_names_L = self.getRainLImageNames(train_data_dir, rain_L_dir, gt_dir)
            self.input_names_H, self.gt_names_H = self.getRainHImageNames(train_data_dir, rain_H_dir, gt_dir)
            self.input_names = self.input_names_L + self.input_names_H 
            self.gt_names = self.gt_names_L + self.gt_names_H 
        ### NOTE: add enchancement training data ###
        # self.training_enhance_percentage = 0.2
        # self.gt_names_enhance = self.gt_names.copy()
        # shuffle(self.gt_names_enhance)
        # self.gt_names_enhance = self.gt_names_enhance[:int(len(self.input_names) * self.training_enhance_percentage)]
        # print("[INFO] len of trainData without enhancement: {}".format(len(self.input_names)))
        # print("[INFO] Training gt data for enhancement: {}".format(len(self.gt_names_enhance)))
        # self.input_names += self.gt_names_enhance
        # self.gt_names += self.gt_names_enhance
        ############################################
        print("Total number of training data: ", len(self.gt_names))
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def getImageNames(self, root_dir, image_dir, gt_dir):
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
            image_ind = re.findall(r'\d+', in_name)[0]
            gt_name = os.path.join(output_dir, image_ind + "_clean.png")
            ### NOTE: choice 2 ###
            # gt_name = in_name.replace(image_dir, gt_dir).replace("Rain_L_", "No_Rain_")
            if os.path.exists(gt_name):
                image_names.append(in_name)
                gt_names.append(gt_name)
        return image_names, gt_names

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
    
    def getCutMixImageNames(self, input_img, gt_img):
        portion = 0.4
        target_idx = random.randint(0, len(self.input_names) - 1)
        target_input_name = self.input_names[target_idx]
        target_gt_name = self.gt_names[target_idx]
        target_input_img = Image.open(target_input_name)
        target_gt_img = Image.open(target_gt_name)
        target_input_img_array = np.array(target_input_img)
        target_gt_img_array = np.array(target_gt_img)
        input_img_array = np.array(input_img)
        gt_img_array = np.array(gt_img)

        CutMix_h, CutMix_w = int(gt_img_array.shape[0] * portion), int(gt_img_array.shape[1] * portion)
        CutMix_start_h, CutMix_start_w = random.randint(0, gt_img_array.shape[0] - CutMix_h), random.randint(0, gt_img_array.shape[1] - CutMix_w)
        while CutMix_start_h + CutMix_h > gt_img_array.shape[0] or CutMix_start_w + CutMix_w > gt_img_array.shape[1] or CutMix_start_h + CutMix_h > target_gt_img_array.shape[0] or CutMix_start_w + CutMix_w > target_gt_img_array.shape[1]:
            CutMix_start_h, CutMix_start_w = random.randint(0, gt_img_array.shape[0] - CutMix_h), random.randint(0, gt_img_array.shape[1] - CutMix_w)
        gt_img_array[CutMix_start_h:CutMix_start_h + CutMix_h, CutMix_start_w:CutMix_start_w + CutMix_w] = target_gt_img_array[CutMix_start_h:CutMix_start_h + CutMix_h, CutMix_start_w:CutMix_start_w + CutMix_w]
        input_img_array[CutMix_start_h:CutMix_start_h + CutMix_h, CutMix_start_w:CutMix_start_w + CutMix_w] = target_input_img_array[CutMix_start_h:CutMix_start_h + CutMix_h, CutMix_start_w:CutMix_start_w + CutMix_w]
        input_img = Image.fromarray(input_img_array)
        gt_img = Image.fromarray(gt_img_array)
        return input_img, gt_img

    def get_images(self, index):
        # --- NOTE: data augmentation --- #
        aug = random.randint(0, 6)
        
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        input_img = Image.open(input_name)

        try:
            gt_img = Image.open(gt_name)
        except:
            gt_img = Image.open(gt_name).convert('RGB')

        # # --- NOTE: data augmentation: CutMix --- #
        # if aug == 1:
        #     input_img, gt_img = self.getCutMixImageNames(input_img, gt_img)

        shift_offset = 10
        input_img_np = np.array(input_img)
        pad_left_np = np.zeros((input_img_np.shape[0], shift_offset, input_img_np.shape[2])).astype(np.uint8)
        input_img_np = np.concatenate((pad_left_np, input_img_np[:, shift_offset:, :]), axis=1)
        input_img = Image.fromarray(input_img_np)

        width, height = input_img.size

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size

        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- TODO: data augmentation --- #
        if aug == 2:
            input_im = TF.hflip(input_im)
            gt = TF.hflip(gt)
        elif aug == 3:
            input_im = TF.vflip(input_im)
            gt = TF.vflip(gt)
        elif aug == 4:
            input_im = TF.rotate(input_im, 90)
            gt = TF.rotate(gt, 90)
        if aug == 5:
            input_im = TF.rotate(input_im, 180)
            gt = TF.rotate(gt, 180)
        elif aug == 6:
            input_im = TF.rotate(input_im, 270)
            gt = TF.rotate(gt, 270)
        # if aug == 2:
        #     input_im = TF.gaussian_blur(input_im, kernel_size=3)
        #     gt = TF.gaussian_blur(gt, kernel_size=3)
        # elif aug == 3:
        #     input_im = TF.gaussian_blur(input_im, kernel_size=5)
        #     gt = TF.gaussian_blur(gt, kernel_size=5)
        # elif aug == 4:
        #     input_im = TF.adjust_brightness(input_im, 0.5)
        #     gt = TF.adjust_brightness(gt, 0.5)
        # elif aug == 5:
        #     input_im = TF.adjust_brightness(input_im, 2)
        #     gt = TF.adjust_brightness(gt, 2)
        # elif aug == 6:
        #     input_im = TF.adjust_contrast(input_im, 0.5)
        #     gt = TF.adjust_contrast(gt, 0.5)
        # elif aug == 7:
        #     input_im = TF.adjust_contrast(input_im, 2)
        #     gt = TF.adjust_contrast(gt, 2)


        # --- Normalize the input image --- #
        normalize_input = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_im = normalize_input(input_im)

        # --- Check the channel is 3 or not --- #
        # if list(input_im.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
        if list(input_im.shape)[0] != 3 or list(gt.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return input_im, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

class TrainData_new(data.Dataset):
    def __init__(self, crop_size, train_data_dir,train_filename):
        super().__init__()
        train_list = train_data_dir + train_filename
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input','gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/',input_name)[-1][:-4]
        
        input_img = Image.open(self.train_data_dir + input_name)


        try:
            gt_img = Image.open(self.train_data_dir + gt_name)
        except:
            gt_img = Image.open(self.train_data_dir + gt_name).convert('RGB')

        width, height = input_img.size
        tmp_ch = 0

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size
        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])

        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        
        # --- Check the channel is 3 or not --- #
        # print(input_im.shape)
        if list(input_im.shape)[0] != 3 or list(gt.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_name))


        return input_im, gt, img_id,R_map,trans_map

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
