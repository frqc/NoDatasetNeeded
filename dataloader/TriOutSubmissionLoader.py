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

    left_fold  = 'right_cam/'
    right_fold = 'left_cam/'


    image = [img for img in os.listdir(filepath+left_fold) if img.find('.png') > -1]


    left_test  = [filepath+left_fold+img for img in image]
    right_test = [filepath+right_fold+img for img in image]

    return left_test, right_test
