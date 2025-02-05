import torch

from collections import OrderedDict
from utils import AverageMeter
from metrics import iou_score


def val(val_loader, model, criterion):

    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'SE': AverageMeter(),
                  'PC': AverageMeter(),
                  'F1': AverageMeter(),
                  'SP': AverageMeter(),
                  'ACC': AverageMeter(),
                  'HD95': AverageMeter()}

    model.eval()

    with torch.no_grad():

        for inputs, masks, _ in val_loader:

            inputs = inputs.cuda()
            masks = masks.cuda()

            outputs = model.forward(inputs)

            loss = criterion(outputs, masks)
            iou, dice, SE, PC, F1, SP, ACC, HD95 = iou_score(outputs, masks)

            avg_meters['loss'].update(loss.item(), inputs.size(0))
            avg_meters['iou'].update(iou, inputs.size(0))
            avg_meters['dice'].update(dice, inputs.size(0))
            avg_meters['SE'].update(SE, inputs.size(0))
            avg_meters['PC'].update(PC, inputs.size(0))
            avg_meters['F1'].update(F1, inputs.size(0))
            avg_meters['SP'].update(SP, inputs.size(0))
            avg_meters['ACC'].update(ACC, inputs.size(0))
            avg_meters['HD95'].update(HD95, inputs.size(0))

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('SE', avg_meters['SE'].avg),
                        ('PC', avg_meters['PC'].avg),
                        ('F1', avg_meters['F1'].avg),
                        ('SP', avg_meters['SP'].avg),
                        ('ACC', avg_meters['ACC'].avg),
                        ('HD95', avg_meters['HD95'].avg)])
