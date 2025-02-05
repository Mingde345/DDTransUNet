import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import losses

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
from adabound import AdaBound

from collections import OrderedDict
from glob import glob

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize, Flip

from sklearn.model_selection import train_test_split

from model.DDTransUNet import DDTransUNet
from train import train
from val import val

from dataset import Dataset


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='0')

    parser.add_argument('-model_name', type=str, default='DDTransUNet', help='model name')
    parser.add_argument('-criterion', type=str, default='BCEDiceLoss')

    parser.add_argument('-epochs', type=int, default=200, help='number of total epochs to run')

    parser.add_argument('-in_channels', type=int, default=3, help='input channels')
    parser.add_argument('-out_channels', type=int, default=1, help='output channels')
    parser.add_argument('-input_size_w', type=int, default=256, help='image width')
    parser.add_argument('-input_size_h', type=int, default=256, help='image height')

    # dataset
    parser.add_argument('-dataset', type=str, default='TN3K', help='dataset name')
    parser.add_argument('-root_dir', type=str, default='/hy-tmp/TN3K/')
    parser.add_argument('-img-ext', default='.jpg', help='image file extension')
    parser.add_argument('-mask-ext', default='.jpg', help='mask file extension')
    parser.add_argument('-batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('-num_workers', type=int, default=0)

    # optimizer
    parser.add_argument('-optimizer', default='Adam', choices=['Adam', 'SGD', 'AdaBound'])
    parser.add_argument('-lr', type=float, default=0.0001, metavar='LR', help='initial learning rate')
    parser.add_argument('-momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-weight_decay', type=float, default=1e-4, help='weight decay')

    # scheduler
    parser.add_argument('-scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('-min_lr', type=float, default=1e-5, help='minimum learning rate')
    parser.add_argument('-factor', type=float, default=0.1)
    parser.add_argument('-patience', type=int, default=2)
    parser.add_argument('-milestones', type=str, default='1,2')
    parser.add_argument('-gamma', type=float, default=2/3)
    parser.add_argument('-early_stopping', type=int, default=-1, metavar='N', help='early stopping (default: -1)')

    return parser.parse_args()


def main():

    config = get_arguments()

    os.makedirs('/hy-tmp/results', exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    # create model

    if config.model_name == 'DDTransUNet':

        model = DDTransUNet(in_chans=config.in_channels, out_chans=config.out_channels)

    else:

        raise NotImplementedError

    torch.cuda.set_device(device=0)
    model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())

    if config.optimizer == 'Adam':

        optimizer = optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)

    elif config.optimzier == 'SGD':

        optimizer = optim.SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    
    elif config.optimzier == 'AdaBound':

        optimizer = AdaBound(params, lr=config.lr, final_lr=0.1)

    else:

        raise NotImplementedError

    if config.scheduler == 'CosineAnnealingLR':

        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)

    elif config.scheduler == 'ReduceLROnPlateau':

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.factor, patience=config.patience,
                                                   verbose=True, min_lr=config.min_lr)

    elif config.scheduler == 'MultiStepLR':

        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config.milestones.split(',')],
                                             gamma=config.gamma)

    elif config.scheduler == 'ConstantLR':

        scheduler = None

    else:

        raise NotImplementedError

    if config.criterion == 'BCEDiceLoss':

        criterion = losses.BCEDiceLoss().cuda()

    else:

        raise NotImplementedError

    cudnn.benchmark = True

    # data loading
    img_ids = glob(os.path.join(config.root_dir, 'image', '*' + config.img_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        RandomRotate90(),
        Flip(),
        Resize(config.input_size_h, config.input_size_w),
        transforms.Normalize()
    ])

    val_transform = Compose([
        Resize(config.input_size_h, config.input_size_w),
        transforms.Normalize()
    ])

    train_data = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config.root_dir, 'image'),
        mask_dir=os.path.join(config.root_dir, 'mask'),
        img_ext=config.img_ext,
        mask_ext=config.mask_ext,
        transform=train_transform)
    
    val_data = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config.root_dir, 'image'),
        mask_dir=os.path.join(config.root_dir, 'mask'),
        img_ext=config.img_ext,
        mask_ext=config.mask_ext,
        transform=val_transform
    )


    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=0)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', [])])

    best_iou = 0
    trigger = 0

    print('Training ' + config.model_name + ' from scratch...')

    for epoch in range(config.epochs):

        print('Starting epoch {}/{}'.format(epoch+1, config.epochs))

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer)

        # validation
        val_log = val(val_loader, model, criterion)

        if config.scheduler == 'CosineAnnealingLR':

            scheduler.step()

        elif config.scheduler == 'ReduceLROnPlateau':

            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_SE %.4f'
              ' - val_PC %.4f - val_F1 %.4f - val_SP %.4f - val_ACC %.4f - val_HD95 %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['dice'],
                 val_log['SE'], val_log['PC'], val_log['F1'], val_log['SP'], val_log['ACC'], val_log['HD95']))

        log['epoch'].append(epoch)
        log['lr'].append(config.lr)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('/hy-tmp/results/log.csv', index=False)

        trigger += 1

        if val_log['iou'] > best_iou:

            torch.save(model.state_dict(), '/hy-tmp/results/model.pth')

            best_iou = val_log['iou']

            print("Saving best model.")

            trigger = 0

        # early stopping

        if 0 <= config.early_stopping <= trigger:

            print("Early stopping")

            break

        print("\n")

        torch.cuda.empty_cache()


if __name__ == "__main__":

    main()
