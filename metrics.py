import numpy as np
import torch

from scipy.ndimage.morphology import distance_transform_edt as edt


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # TP : True Positive
    # FN : False Negative
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
    return SE


def get_specificity(SR, GT, threshold=0.5):
    SP = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
    return SP


def get_precision(SR, GT, threshold=0.5):
    PC = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)
    return PC


def iou_score(output, target):

    smooth = 1e-5

    hd_metric = HausdorffDistance()

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)

    output_ = torch.tensor(output_)
    target_ = torch.tensor(target_)
    SE = get_sensitivity(output_, target_, threshold=0.5)
    PC = get_precision(output_, target_, threshold=0.5)
    SP = get_specificity(output_, target_, threshold=0.5)
    ACC = get_accuracy(output_, target_, threshold=0.5)
    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    hd95 = hd_metric.compute(output, target)

    return iou, dice, SE, PC, F1, SP, ACC, hd95


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.percentile(distances[indexes], 95))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
            ), "Only binary channel supported"

        pred = (pred > 0.5).astype(np.byte())
        target = (target > 0.5).astype(np.byte())

        if np.sum(pred) == 0:

            pred[0][0][0][0] = 1

        pred_tensor = torch.from_numpy(pred).float()
        target_tensor = torch.from_numpy(target).float()

        right_hd = torch.from_numpy(
            self.hd_distance(pred_tensor.cpu().numpy(), target_tensor.cpu().numpy())
        ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target_tensor.cpu().numpy(), pred_tensor.cpu().numpy())
        ).float()

        return torch.max(right_hd, left_hd)
