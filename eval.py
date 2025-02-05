import argparse
import os

import numpy as np
import pandas as pd
import torch
import cv2

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from glob import glob

from dataset import Dataset
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import Resize

from metrics import iou_score

from model.DDTransUNet import DDTransUNet

from collections import OrderedDict

from utils import AverageMeter


def get_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-gpu', type=str, default='0')

    parser.add_argument('-model_name', type=str, default='DDTransUNet')

    parser.add_argument('-in_channels', type=int, default=3, help='input channels')
    parser.add_argument('-out_channels', type=int, default=1, help='output channels')
    parser.add_argument('-input_size_w', type=int, default=256, help='image width')
    parser.add_argument('-input_size_h', type=int, default=256, help='image height')

    parser.add_argument('-root_dir', type=str, default='F:/Eval/TN3K/')
    parser.add_argument('-img-ext', default='.jpg', help='image file extension')
    parser.add_argument('-mask-ext', default='.jpg', help='mask file extension')
    parser.add_argument('-load_path', type=str, default='F:/Eval/TN3K/weight/model.pth')
    parser.add_argument('-save_dir', type=str, default='F:/Eval/TN3K/mask/')
    parser.add_argument('-test_dataset', type=str, default='TN3K')

    return parser.parse_args()


def main():

    config = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    if 'DDTransUNet' == config.model_name:

        model = DDTransUNet(in_chans=config.in_channels, out_chans=config.out_channels)

    else:

        raise NotImplementedError

    model.load_state_dict(torch.load(config.load_path))
    model.cuda()

    test_img_ids = glob(os.path.join(config.root_dir, 'test_image', '*' + config.img_ext))
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]

    test_transform = Compose([
        Resize(config.input_size_h, config.input_size_w),
        transforms.Normalize()
    ])

    if config.test_dataset == 'TN3K':
        
        test_data = Dataset(
            img_ids=test_img_ids,
            img_dir=os.path.join(config.root_dir, 'test_image'),
            mask_dir=os.path.join(config.root_dir, 'test_mask'),
            img_ext=config.img_ext,
            mask_ext=config.mask_ext,
            transform=test_transform
        )
    
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    if not os.path.exists(config.save_dir):
        
        os.makedirs(config.save_dir)

    avg_meters = {'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'SE': AverageMeter(),
                  'PC': AverageMeter(),
                  'F1': AverageMeter(),
                  'SP': AverageMeter(),
                  'ACC': AverageMeter(),
                  'HD95': AverageMeter()}

    log = OrderedDict([
        ('val_iou', []),
        ('val_dice', []),
        ('val_ACC', []),
        ('val_HD95', [])])
    
    model.eval()
    
    with torch.no_grad():
        
        for inputs, masks, meta in tqdm(test_loader, total=len(test_loader)):

            inputs = inputs.cuda()
            masks = masks.cuda()
            model = model.cuda()
            
            outputs = model.forward(inputs)

            iou, dice, SE, PC, F1, SP, ACC, HD95 = iou_score(outputs, masks)

            log['val_iou'].append(iou)
            log['val_dice'].append(dice)
            log['val_ACC'].append(ACC)
            log['val_HD95'].append(HD95)

            pd.DataFrame(log).to_csv('F:/Work/log.csv', index=False)

            avg_meters['iou'].update(iou, inputs.size(0))
            avg_meters['dice'].update(dice, inputs.size(0))
            avg_meters['SE'].update(SE, inputs.size(0))
            avg_meters['PC'].update(PC, inputs.size(0))
            avg_meters['F1'].update(F1, inputs.size(0))
            avg_meters['SP'].update(SP, inputs.size(0))
            avg_meters['ACC'].update(ACC, inputs.size(0))
            avg_meters['HD95'].update(HD95, inputs.size(0))

            outputs = torch.sigmoid(outputs).cpu().numpy()
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0

            for i in range(len(outputs)):

                cv2.imwrite(os.path.join(config.save_dir, meta['img_id'][i] + '.png'), (outputs[i, 0] * 255).astype(np.uint8))

    test_log = OrderedDict([('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg),
                            ('SE', avg_meters['SE'].avg),
                            ('PC', avg_meters['PC'].avg),
                            ('F1', avg_meters['F1'].avg),
                            ('SP', avg_meters['SP'].avg),
                            ('ACC', avg_meters['ACC'].avg),
                            ('HD95', avg_meters['HD95'].avg)])

    print("Test Result:")
    print('test_iou %.4f - test_dice %.4f - test_SE %.4f - test_PC %.4f - test_F1 %.4f - test_SP %.4f - test_ACC %.4f'
          ' - test_HD95 %.4f'
          % (test_log['iou'], test_log['dice'], test_log['SE'], test_log['PC'], test_log['F1'], test_log['SP'],
             test_log['ACC'], test_log['HD95']))
    
    torch.cuda.empty_cache()


if __name__ == '__main__':

    main()
