import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAngularMargin(nn.Module):
    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        """The Implementation of Additive Angular Margin (AAM) proposed
       in the following paper: '''Margin Matters: Towards More Discriminative Deep Neural Network Embeddings for Speaker Recognition'''
       (https://arxiv.org/abs/1906.07317)

        Args:
            margin (float, optional): margin factor. Defaults to 0.0.
            scale (float, optional): scale factor. Defaults to 1.0.
            easy_margin (bool, optional): easy_margin flag. Defaults to False.
        """
        super(AdditiveAngularMargin, self).__init__()
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, outputs, targets):
        cosine = outputs.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs


class AAMLoss(nn.Module):
    def __init__(self, margin=0.2, scale=30, easy_margin=False):
        super(AAMLoss, self).__init__()
        self.loss_fn = AdditiveAngularMargin(margin=margin, scale=scale, easy_margin=easy_margin)
        self.criterion = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets):
        targets = F.one_hot(targets, outputs.shape[1]).float()
        predictions = self.loss_fn(outputs, targets)
        predictions = F.log_softmax(predictions, dim=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss


class AMLoss(nn.Module):
    def __init__(self, margin=0.2, scale=30):
        super(AMLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, outputs, targets):
        label_view = targets.view(-1, 1)
        delt_costh = torch.zeros(outputs.size(), device=targets.device).scatter_(1, label_view, self.m)
        costh_m = outputs - delt_costh
        predictions = self.s * costh_m
        loss = self.criterion(predictions, targets) / targets.shape[0]
        return loss


class ARMLoss(nn.Module):
    def __init__(self, margin=0.2, scale=30):
        super(ARMLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, outputs, targets):
        label_view = targets.view(-1, 1)
        delt_costh = torch.zeros(outputs.size(), device=targets.device).scatter_(1, label_view, self.m)
        costh_m = outputs - delt_costh
        costh_m_s = self.s * costh_m
        delt_costh_m_s = costh_m_s.gather(1, label_view).repeat(1, costh_m_s.size()[1])
        costh_m_s_reduct = costh_m_s - delt_costh_m_s
        predictions = torch.where(costh_m_s_reduct < 0.0, torch.zeros_like(costh_m_s), costh_m_s)
        loss = self.criterion(predictions, targets) / targets.shape[0]
        return loss


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, outputs, targets):
        loss = self.criterion(outputs, targets) / targets.shape[0]
        return loss
