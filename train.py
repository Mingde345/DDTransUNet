import torch

from collections import OrderedDict
from utils import AverageMeter
from metrics import iou_score


def train(train_loader, model, criterion, optimizer):

    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    for inputs, masks, _ in train_loader:

        inputs = inputs.cuda()
        masks = masks.cuda()

        outputs = model.forward(inputs)

        loss = criterion(outputs, masks)

        iou, _, _, _, _, _, _, _ = iou_score(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), inputs.size(0))
        avg_meters['iou'].update(iou, inputs.size(0))

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
