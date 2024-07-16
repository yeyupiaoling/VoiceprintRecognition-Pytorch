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
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(torch.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        sine = torch.sqrt(1.0 - torch.pow(logits, 2))
        phi = logits * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(logits > 0, phi, logits)
        else:
            phi = torch.where(logits > self.th, phi, logits - self.mmm)

        one_hot = torch.zeros(logits.size()).type_as(logits)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * logits)
        output *= self.scale

        loss = self.criterion(output, labels)
        return loss

    def update(self, margin=0.2):
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)


class SphereFace2(nn.Module):
    def __init__(self, margin=0.2, scale=32.0, lanbuda=0.7, t=3, margin_type='C'):
        """Implement of sphereface2 for speaker verification:
            Reference:
                [1] Exploring Binary Classification Loss for Speaker Verification
                https://ieeexplore.ieee.org/abstract/document/10094954
                [2] Sphereface2: Binary classification is all you need for deep face recognition
                https://arxiv.org/pdf/2108.01513
            Args:
                scale: norm of input feature
                margin: margin
                lanbuda: weight of positive and negative pairs
                t: parameter for adjust score distribution
                margin_type: A:cos(theta+margin) or C:cos(theta)-margin
            Recommend margin:
                training: 0.2 for C and 0.15 for A
                LMF: 0.3 for C and 0.25 for A
        """
        super(SphereFace2, self).__init__()
        self.scale = scale
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.t = t
        self.lanbuda = lanbuda
        self.margin_type = margin_type

        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)

    def fun_g(self, z, t: int):
        gz = 2 * torch.pow((z + 1) / 2, t) - 1
        return gz

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(torch.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        if self.margin_type == 'A':  # arcface type
            sin = torch.sqrt(1.0 - torch.pow(logits, 2))
            cos_m_theta_p = self.scale * self.fun_g(
                torch.where(logits > self.th, logits * self.cos_m - sin * self.sin_m, logits - self.mmm), self.t) + \
                            self.bias[0][0]
            cos_m_theta_n = self.scale * self.fun_g(logits * self.cos_m + sin * self.sin_m, self.t) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))
        else:
            # cosface type
            cos_m_theta_p = self.scale * (self.fun_g(logits, self.t) - self.margin) + self.bias[0][0]
            cos_m_theta_n = self.scale * (self.fun_g(logits, self.t) + self.margin) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))

        target_mask = torch.zeros(logits.size()).type_as(logits)
        target_mask.scatter_(1, labels.view(-1, 1).long(), 1.0)
        nontarget_mask = 1 - target_mask
        loss = (target_mask * cos_p_theta + nontarget_mask * cos_n_theta).sum(1).mean()
        return loss

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)


class AMLoss(nn.Module):
    def __init__(self, margin=0.3, scale=32):
        super(AMLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

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


class ARMLoss(nn.Module):
    def __init__(self, margin=0.3, scale=32):
        super(ARMLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

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
        costh_m_s = self.scale * costh_m
        delt_costh_m_s = costh_m_s.gather(1, label_view).repeat(1, costh_m_s.size()[1])
        costh_m_s_reduct = costh_m_s - delt_costh_m_s
        predictions = torch.where(costh_m_s_reduct < 0.0, torch.zeros_like(costh_m_s), costh_m_s)
        loss = self.criterion(predictions, labels) / labels.shape[0]
        return loss

    def update(self, margin=0.2):
        self.margin = margin


class CELoss(nn.Module):
    def __init__(self, **kwargs):
        super(CELoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(torch.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        loss = self.criterion(logits, labels) / labels.shape[0]
        return loss

    def update(self, margin=0.2):
        pass


class SubCenterLoss(nn.Module):
    r"""Implement of large margin arc distance with subcenter:
    Reference:Sub-center ArcFace: Boosting Face Recognition byLarge-Scale Noisy
     Web Faces.https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf

     Args:
        margin (float, optional): margin factor. Defaults to 0.3.
        scale (float, optional): scale factor. Defaults to 32.0.
        easy_margin (bool, optional): easy_margin flag. Defaults to False.
        K: number of sub-centers, same classifier K.
    """

    def __init__(self, margin=0.2, scale=32, easy_margin=False, K=3):
        super(SubCenterLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        # subcenter
        self.K = K
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        """
        Args:
            inputs(dict): 模型输出的特征向量 (batch_size, feat_dim) 和分类层输出的logits(batch_size, class_num)
            labels(torch.Tensor): 类别标签 (batch_size)
        """
        features, logits = inputs['features'], inputs['logits']
        # (batch, out_dim, k)
        cosine = torch.reshape(logits, (-1, logits.shape[1] // self.K, self.K))
        # (batch, out_dim)
        cosine, _ = torch.max(cosine, 2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)

        one_hot = logits.new_zeros(cosine.size())
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = self.criterion(output, labels)
        return loss

    def update(self, margin=0.2):
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)


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
                 an_value=0.4):
        super(TripletAngularMarginLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)
        self.normalize_feature = normalize_feature
        self.add_absolute = add_absolute
        self.ap_value = ap_value
        self.an_value = an_value
        self.absolute_loss_weight = absolute_loss_weight
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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
