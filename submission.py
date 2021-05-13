from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
from PIL import Image

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar',
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.KITTI == '2015':
   from dataloader import KITTI_submission_loader as DA
elif args.KITTI == 'triout':
    from dataloader import TriOutSubmissionLoader as DA
else:
   from dataloader import KITTI_submission_loader2012 as DA  

test_left_img, test_right_img = DA.dataloader(args.datapath)

if args.model == 'stackhourglass':
    model = stackhourglass(int(args.maxdisp))
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
    model.eval()

    if args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()     

    with torch.no_grad():
        output = model(imgL,imgR)
    output = torch.squeeze(output).data.cpu().numpy()
    return output

def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor()
        #, transforms.Normalize(**normal_mean_var)
        ])    

    for inx in range(len(test_left_img)):

        imgL_o = Image.open(test_left_img[inx]).convert('RGB')
        imgR_o = Image.open(test_right_img[inx]).convert('RGB')

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)         

        # # TODO: find a proper way to crop image
        imgL = imgL[:, ::2, ::2]
        imgR = imgR[:, ::2, ::2]

        print(imgR.shape)
        print(imgL)
        print(imgR)

        print('imgL.shape ', imgL.numpy().shape)
        transforms.ToPILImage()(imgL).save(test_left_img[inx].split('/')[-1]+'.left.png')
        transforms.ToPILImage()(imgR).save(test_left_img[inx].split('/')[-1]+'.right.png')

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0    


        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)


        start_time = time.time()
        pred_disp = test(imgL,imgR)
        print('pred_disp shape: ', pred_disp.shape)
        print('time = %.2f' %(time.time() - start_time))

        if right_pad != 0:
            img = pred_disp[top_pad:,:-right_pad]
        elif top_pad !=0:
            img = pred_disp[top_pad:,:]
        else:
            img = pred_disp

        print('img shape: ', img.shape)
        img = (img*256).astype('uint16')
        print(img)
        print('img shape: ', img.shape)
        img = Image.fromarray(img)
        print('img: ', img)
        print('path: ', test_left_img[inx].split('/')[-1])
        img.convert('L').save(test_left_img[inx].split('/')[-1])


if __name__ == '__main__':
    main()






