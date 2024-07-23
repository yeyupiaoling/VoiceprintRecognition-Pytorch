import torch
import torch.nn as nn


class AMLoss(nn.Module):
    def __init__(self, margin=0.3, scale=32, label_smoothing=0.0):
        super(AMLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum", label_smoothing=label_smoothing)

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(torch.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        label_view = labels.view(-1, 1)
        delt_costh = torch.zeros(logits.size(), device=labels.device).scatter_(1, label_view, self.margin)
        costh_m = logits - delt_costh
        predictions = self.scale * costh_m
        loss = self.criterion(predictions, labels) / labels.shape[0]
        return loss

    def update(self, margin=0.2):
        self.margin = margin
