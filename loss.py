import torch.nn as nn
import torch
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = inputs.sum() + targets.sum()
        dice_coef = (2. * intersection + smooth) / (total + smooth)
        dice_loss = 1 - dice_coef
        return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):

        dice_loss = self.dice_loss(inputs, targets)
        # targets = targets.squeeze(1)
        # targets = torch.round(targets).to(torch.long)
        bce_loss = self.bce_loss(inputs, targets)
        return 0.5*dice_loss + 0.5*bce_loss


def convert_to_one_hot(mask, num_classes=2):
    mask = mask.to(torch.int64)

    one_hot = F.one_hot(mask, num_classes=num_classes)
    one_hot = one_hot.squeeze(1)


    one_hot = one_hot.permute(0, 4, 1, 2, 3)
    one_hot = one_hot.to(float)

    return one_hot

