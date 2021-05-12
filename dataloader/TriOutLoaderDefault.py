import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

    right_fold  = 'left_1/'
    left_fold = 'right_cam/'
    disp_L = 'depth/'
    # disp_R = 'disp_occ_1/'

    image = [img for img in os.listdir(filepath+left_fold) if img.find('.png') > -1]

    train = image[:70]
    val   = image[70:]    

    left_train  = [filepath+left_fold+img for img in train]
    right_train = [filepath+right_fold+img for img in train]
    disp_train_L = [filepath+disp_L+img for img in train]

    left_val  = [filepath+left_fold+img for img in val]
    right_val = [filepath+right_fold+img for img in val]
    disp_val_L = [filepath+disp_L+img for img in val]

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L