import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcNet(nn.Module):
    def __init__(self,
                 feature_dim,
                 class_dim,
                 margin=0.2,
                 scale=30.0,
                 easy_margin=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.class_dim = class_dim
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        self.weight = Parameter(torch.FloatTensor(feature_dim, class_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        input_norm = torch.sqrt(torch.sum(torch.square(input), dim=1, keepdim=True))
        input = torch.divide(input, input_norm)

        weight_norm = torch.sqrt(torch.sum(torch.square(self.weight), dim=0, keepdim=True))
        weight = torch.divide(self.weight, weight_norm)

        cos = torch.matmul(input, weight)
        sin = torch.sqrt(1.0 - torch.square(cos) + 1e-6)
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        phi = cos * cos_m - sin * sin_m

        th = math.cos(self.margin) * (-1)
        mm = math.sin(self.margin) * self.margin
        if self.easy_margin:
            phi = self._paddle_where_more_than(cos, 0, phi, cos)
        else:
            phi = self._paddle_where_more_than(cos, th, phi, cos - mm)
        one_hot = torch.nn.functional.one_hot(label, self.class_dim)
        one_hot = torch.squeeze(one_hot, dim=1)
        output = torch.multiply(one_hot, phi) + torch.multiply((1.0 - one_hot), cos)
        output = output * self.scale
        return output

    def _paddle_where_more_than(self, target, limit, x, y):
        mask = (target > limit).float()
        output = torch.multiply(mask, x) + torch.multiply((1.0 - mask), y)
        return output
