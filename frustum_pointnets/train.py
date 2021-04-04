import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_provider import FrustumDataset, compute_box3d_iou
from frustum_pointnet import FrustumPointNetv1
from model_util import FrustumPointNetLoss

LOG_FOUT = open(os.path.join(os.getcwd(), 'log_train.txt'), 'w')
MAX_EPOCH = 100


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train():
    point_net = FrustumPointNetv1().cuda().train()
    loss = FrustumPointNetLoss()
    optimizer = torch.optim.Adam(point_net.parameters())

    # record for one epoch
    train_iou2d = 0.0
    train_iou3d = 0.0
    train_iou3d_acc = 0.0

    for epoch in range(MAX_EPOCH):
        log_string('**** EPOCH %03d ****' % (epoch + 1))
        sys.stdout.flush()
        log_string('Epoch %d/%s:' % (epoch + 1, MAX_EPOCH))

        train_acc = 0
        n_samples = 0

        for data_index, data in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):

            batch_data, batch_label, batch_center, \
            batch_hclass, batch_hres, \
            batch_sclass, batch_sres, \
            batch_rot_angle, batch_one_hot_vec = [d.float().cuda() for d in data]

            logits, mask, stage1_center, center_boxnet, \
            heading_scores, heading_residuals_normalized, heading_residuals, \
            size_scores, size_residuals_normalized, size_residuals, center = \
                point_net(batch_data, batch_one_hot_vec)

            total_loss = loss(logits, batch_label,
                              center, batch_center, stage1_center,
                              heading_scores, heading_residuals_normalized,
                              heading_residuals,
                              batch_hclass, batch_hres,
                              size_scores, size_residuals_normalized,
                              size_residuals,
                              batch_sclass, batch_sres)

            total_loss.backward()
            optimizer.step()

            iou2ds, iou3ds = compute_box3d_iou(
                center.cpu().detach().numpy(),
                heading_scores.cpu().detach().numpy(),
                heading_residuals.cpu().detach().numpy(),
                size_scores.cpu().detach().numpy(),
                size_residuals.cpu().detach().numpy(),
                batch_center.cpu().detach().numpy(),
                batch_hclass.cpu().detach().numpy(),
                batch_hres.cpu().detach().numpy(),
                batch_sclass.cpu().detach().numpy(),
                batch_sres.cpu().detach().numpy())

            train_iou2d += np.sum(iou2ds)
            train_iou3d += np.sum(iou3ds)
            train_iou3d_acc += np.sum(iou3ds >= 0.7)

            correct = torch.argmax(logits, 2).eq(batch_label.long()).detach().cpu().numpy()

            train_acc += np.sum(correct)
            n_samples += data[0].shape[0]

        train_acc /= n_samples * float(512)
        print("Correct: " + str(train_acc))


if __name__ == '__main__':
    input_file = "/opt/data/frustum_kitti.pickle"
    dataset = FrustumDataset(input_file, 512)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)

    train()

    # TODO: add more metrics and logging
    # TODO: add breakup export and import
    # TODO: reorganize function input
