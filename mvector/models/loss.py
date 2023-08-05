import math

import torch
import torch.nn as nn


class AAMLoss(nn.Module):
    def __init__(self, margin=0.2, scale=32, easy_margin=False):
        """The Implementation of Additive Angular Margin (AAM) proposed
       in the following paper: '''Margin Matters: Towards More Discriminative Deep Neural Network Embeddings for Speaker Recognition'''
       (https://arxiv.org/abs/1906.07317)

        Args:
            margin (float, optional): margin factor. Defaults to 0.3.
            scale (float, optional): scale factor. Defaults to 32.0.
            easy_margin (bool, optional): easy_margin flag. Defaults to False.
        """
        super(AAMLoss, self).__init__()
        self.scale = scale
        self.easy_margin = easy_margin
        self.criterion = nn.CrossEntropyLoss()

        self.update(margin)

    def forward(self, cosine, label):
        """
        Args:
            cosine (torch.Tensor): cosine distance between the two tensors, shape [batch, num_classes].
            label (torch.Tensor): label of speaker id, shape [batch, ].

        Returns:
            torch.Tensor: loss value.
        """
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)

        one_hot = torch.zeros(cosine.size()).type_as(cosine)
        one_hot.scatter_(1, label.unsqueeze(1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = self.criterion(output, label)
        return loss

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)


class AMLoss(nn.Module):
    def __init__(self, margin=0.3, scale=32):
        super(AMLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, cosine, label):
        label_view = label.view(-1, 1)
        delt_costh = torch.zeros(cosine.size(), device=label.device).scatter_(1, label_view, self.m)
        costh_m = cosine - delt_costh
        predictions = self.s * costh_m
        loss = self.criterion(predictions, label) / label.shape[0]
        return loss

    def update(self, margin=0.2):
        self.margin = margin


class ARMLoss(nn.Module):
    def __init__(self, margin=0.3, scale=32):
        super(ARMLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, cosine, label):
        label_view = label.view(-1, 1)
        delt_costh = torch.zeros(cosine.size(), device=label.device).scatter_(1, label_view, self.m)
        costh_m = cosine - delt_costh
        costh_m_s = self.s * costh_m
        delt_costh_m_s = costh_m_s.gather(1, label_view).repeat(1, costh_m_s.size()[1])
        costh_m_s_reduct = costh_m_s - delt_costh_m_s
        predictions = torch.where(costh_m_s_reduct < 0.0, torch.zeros_like(costh_m_s), costh_m_s)
        loss = self.criterion(predictions, label) / label.shape[0]
        return loss

    def update(self, margin=0.2):
        self.m = margin


class CELoss(nn.Module):
    def __init__(self, **kwargs):
        super(CELoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, cosine, label):
        loss = self.criterion(cosine, label) / label.shape[0]
        return loss

    def update(self, margin=0.2):
        pass
