"""LightUNet：1 通道输入、3 类输出的轻量 U-Net（224x224）。

设计目标：398 例规模数据集 + 低标注场景下不易过拟合，参数量约 2M。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout, dropout=0.0):
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class LightUNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=3, base=32, dropout=0.0):
        super().__init__()
        self.enc1 = _block(in_ch, base, dropout)          # 224
        self.enc2 = _block(base, base * 2, dropout)       # 112
        self.enc3 = _block(base * 2, base * 4, dropout)   # 56
        self.bott = _block(base * 4, base * 8, dropout)   # 28
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _block(base * 8, base * 4, dropout)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _block(base * 4, base * 2, dropout)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _block(base * 2, base, dropout)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def dice_loss(logits, target, num_classes=3, ignore_index=None, eps=1e-6):
    """多类 soft Dice loss（对全部类取平均；被 ignore 的像素不参与）。"""
    probs = F.softmax(logits, dim=1)
    if ignore_index is not None:
        valid = (target != ignore_index)
        target = target.clone()
        target[~valid] = 0
        valid_f = valid.float().unsqueeze(1)
    else:
        valid_f = torch.ones_like(probs[:, :1])
    onehot = F.one_hot(target.clamp(0, num_classes - 1), num_classes)  # (B,H,W,C)
    onehot = onehot.permute(0, 3, 1, 2).float()
    probs = probs * valid_f
    onehot = onehot * valid_f
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()
