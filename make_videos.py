import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

img_dir = "/home/za/Documents/TransWeather_newAdvanture/imgs"


sub_dirs = glob(os.path.join(img_dir, "*"))

for sub_dir in sub_dirs:
    print("----> ", sub_dir)
    img_names = glob(os.path.join(sub_dir, "*.png"))
    img_names.sort()
    sample_img = cv2.imread(img_names[0])
    video = cv2.VideoWriter(os.path.join(img_dir, '{}.avi'.format(sub_dir.split("/")[-1])), cv2.VideoWriter_fourcc(*'XVID'), 30, (sample_img.shape[1], sample_img.shape[0]))
    for i in tqdm(range(len(img_names))):
        img = cv2.imread(img_names[i])
        video.write(img)
    video.release()

