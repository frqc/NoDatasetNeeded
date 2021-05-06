import os
import click
import argparse
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from dataloader import KITTILoader as DA

from models import *


def train(model, optimizer, left_img, right_img, disp_img, device):
    model.train()

    left_img = Variable(torch.FloatTensor(left_img)).to(device)
    right_img = Variable(torch.FloatTensor(right_img)).to(device)
    disp_img = Variable(torch.FloatTensor(disp_img)).to(device)

    optimizer.zero_grad()

    output1, output2, output3 = model(left_img, right_img)

    output1 = torch.squeeze(output1, 1)
    output2 = torch.squeeze(output2, 1)
    output3 = torch.squeeze(output3, 1)

    loss1 = F.smooth_l1_loss(
        output1, disp_img, reduction='mean')
    loss2 = F.smooth_l1_loss(
        output2, disp_img, reduction='mean')
    loss3 = F.smooth_l1_loss(
        output3, disp_img, reduction='mean')

    loss = 0.5 * loss1 + 0.7*loss2 + loss3

    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, left_img, right_img, disp_true, device, scheduler=None):
    model.eval()
    left_img = Variable(torch.FloatTensor(left_img)).to(device)
    right_img = Variable(torch.FloatTensor(right_img)).to(device)

    with torch.no_grad():
        output3 = model(left_img, right_img)

    pred_disp = output3.data.cpu()

    #computing 3-px error#
    delta_disp = np.abs(disp_true-pred_disp)
    correct = (delta_disp < 3) | (delta_disp < disp_true * 0.05)

    torch.cuda.empty_cache()
    total_correct = torch.sum(correct)
    total_pixels = correct.shape[0] * correct.shape[1] * \
        correct.shape[2] * correct.shape[3]

    error_rate = 1 - float(total_correct / total_pixels)

    if scheduler is not None:
        scheduler.step(error_rate)

    return error_rate


@click.command()
@click.option('-g', '--gpu', default=0)
@click.option('-b', '--batch', default=1)
@click.option('-e', '--epochs', default=1)
@click.option('-l', '--left_index', default=1)
@click.option('-r', '--lr', default=0.001)
@click.option('--maxdisp', default=256)
@click.option('--datatype', default='carla')
@click.option('--loadmodel', default='./pretrained_model_KITTI2015.tar')
@click.option('--savemodel', default=None)
@click.option('--seed', default=1)
def main(gpu, batch, left_index, lr, maxdisp, datatype, epochs, loadmodel, savemodel, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        dev = f'cuda:{gpu}'
    else:
        dev = 'cpu'
    device = torch.device(dev)
    print(
        f'Device: {device}, Batch: {batch}, Left index: {left_index}, Learning rate:{lr}')

    if savemodel is None:
        savemodel = "./saved_model_" + str(left_index)
        if not os.path.exists(savemodel):
            os.makedirs(savemodel)

    # [0] Load data
    if datatype == 'carla':
        from dataloader import carla_loader as ls
        datapath = "/home/tim/script/carla_script/_six_out/"
    else:
        from dataloader import KITTIloader2015 as ls
        datapath = "/home/tim/data/training/"

    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
        datapath, left_index)

    TrainImgLoader = torch.utils.data.DataLoader(DA.myImageFloder(
        all_left_img, all_right_img, all_left_disp, True), batch_size=batch, shuffle=True, num_workers=8, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(DA.myImageFloder(
        test_left_img, test_right_img, test_left_disp, False), batch_size=batch, shuffle=False, num_workers=4, drop_last=False)

    # [1] Build model
    model = stackhourglass(maxdisp)
    if loadmodel is not None:
        if loadmodel[-8:] == '2015.tar':
            state_dict = torch.load(loadmodel)
            state_dict = {".".join(k.split(".")[1:]): v for k,
                          v in state_dict['state_dict'].items()}
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict['state_dict'])
        print("loaded", loadmodel)
    model = model.to(device)

    # Todo: Add number of unfrozen parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # [2] Training
    tuning_start_time = time.time()
    max_acc = 0
    for epoch in range(1, epochs+1):

        ## Training ##
        total_train_loss = 0
        for _, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(tqdm(TrainImgLoader)):
            loss = train(model, optimizer, imgL_crop,
                         imgR_crop, disp_crop_L, device)
            total_train_loss += loss
        total_train_loss /= len(TrainImgLoader)

        ## Test ##
        total_test_loss = 0
        for _, (imgL, imgR, disp_L) in enumerate(tqdm(TestImgLoader)):
            test_loss = test(model, imgL, imgR, disp_L, device, scheduler)
            total_test_loss += test_loss
        total_test_loss /= len(TestImgLoader)

        print('epoch %d Training Loss = %.3f, Test 3-px Error Rate = %.3f' %
              (epoch, total_train_loss, total_test_loss*100))

        if total_test_loss*100 > max_acc:
            max_acc = total_test_loss*100
            max_epo = epoch

        print('MAX epoch %d total test error = %.3f' % (max_epo, max_acc))

        # SAVE
        savefilename = savemodel + 'finetune_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss/len(TrainImgLoader),
            'test_loss': total_test_loss*100,
        }, savefilename)

    print('full finetune time = %.2f HR' %
          ((time.time() - tuning_start_time)/3600))
    print(max_epo)
    print(max_acc)


if __name__ == '__main__':
    main()
