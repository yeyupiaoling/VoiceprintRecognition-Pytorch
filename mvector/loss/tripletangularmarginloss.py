import torch
import torch.nn as nn


class TripletAngularMarginLoss(nn.Module):
    """A more robust triplet loss with hard positive/negative mining on angular margin instead of relative distance between d(a,p) and d(a,n).

    Args:
        margin (float, optional): angular margin. Defaults to 0.5.
        normalize_feature (bool, optional): whether to apply L2-norm in feature before computing distance(cos-similarity). Defaults to True.
        add_absolute (bool, optional): whether add absolute loss within d(a,p) or d(a,n). Defaults to True.
        absolute_loss_weight (float, optional): weight for absolute loss. Defaults to 1.0.
        ap_value (float, optional): weight for d(a, p). Defaults to 0.8.
        an_value (float, optional): weight for d(a, n). Defaults to 0.4.
    """

    def __init__(self,
                 margin=0.5,
                 normalize_feature=True,
                 add_absolute=True,
                 absolute_loss_weight=1.0,
                 ap_value=0.8,
                 an_value=0.4,
                 label_smoothing=0.0):
        super(TripletAngularMarginLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)
        self.normalize_feature = normalize_feature
        self.add_absolute = add_absolute
        self.ap_value = ap_value
        self.an_value = an_value
        self.absolute_loss_weight = absolute_loss_weight
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(torch.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        loss_ce = self.criterion(logits, labels)

        if self.normalize_feature:
            features = torch.divide(features, torch.norm(features, p=2, dim=-1, keepdim=True))

        bs = features.size(0)

        # compute distance(cos-similarity)
        dist = torch.matmul(features, features.t())

        # hard negative mining
        is_pos = labels.expand(bs, bs).eq(labels.expand(bs, bs).t())
        is_neg = labels.expand(bs, bs).ne(labels.expand(bs, bs).t())

        # `dist_ap` means distance(anchor, positive)
        dist_ap = dist[is_pos].view(bs, -1).min(dim=1, keepdim=True)[0]
        # `dist_an` means distance(anchor, negative)
        dist_an = dist[is_neg].view(bs, -1).max(dim=1, keepdim=True)[0]
        # shape [N]
        dist_ap = torch.squeeze(dist_ap, dim=1)
        dist_an = torch.squeeze(dist_an, dim=1)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_ap, dist_an, y)

        if self.add_absolute:
            absolut_loss_ap = self.ap_value - dist_ap
            absolut_loss_ap = torch.where(absolut_loss_ap > 0, absolut_loss_ap, torch.zeros_like(absolut_loss_ap))

            absolut_loss_an = dist_an - self.an_value
            absolut_loss_an = torch.where(absolut_loss_an > 0, absolut_loss_an, torch.ones_like(absolut_loss_an))

            loss = (absolut_loss_an.mean() + absolut_loss_ap.mean()) * self.absolute_loss_weight + loss.mean()
        loss = loss + loss_ce
        return loss

    def update(self, margin=0.5):
        self.ranking_loss.margin = margin
