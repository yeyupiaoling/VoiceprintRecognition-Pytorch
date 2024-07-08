import math

import torch
import torch.nn as nn

import torch.nn.functional as F


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

        self.update(margin)

    def fun_g(self, z, t: int):
        gz = 2 * torch.pow((z + 1) / 2, t) - 1
        return gz

    def forward(self, cosine, label):
        """
        Args:
            cosine (torch.Tensor): cosine distance between the two tensors, shape [batch, num_classes].
            label (torch.Tensor): label of speaker id, shape [batch, ].

        Returns:
            torch.Tensor: loss value.
        """
        if self.margin_type == 'A':  # arcface type
            sin = torch.sqrt(1.0 - torch.pow(cosine, 2))
            cos_m_theta_p = self.scale * self.fun_g(
                torch.where(cosine > self.th, cosine * self.cos_m - sin * self.sin_m, cosine - self.mmm), self.t) + \
                            self.bias[0][0]
            cos_m_theta_n = self.scale * self.fun_g(cosine * self.cos_m + sin * self.sin_m, self.t) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))
        else:
            # cosface type
            cos_m_theta_p = self.scale * (self.fun_g(cosine, self.t) - self.margin) + self.bias[0][0]
            cos_m_theta_n = self.scale * (self.fun_g(cosine, self.t) + self.margin) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))

        target_mask = torch.zeros(cosine.size()).type_as(cosine)
        target_mask.scatter_(1, label.view(-1, 1).long(), 1.0)
        nontarget_mask = 1 - target_mask
        loss = (target_mask * cos_p_theta + nontarget_mask * cos_n_theta).sum(1).mean()
        return loss

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)
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
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(math.pi - margin)
        self.m = self.margin
        self.criterion = nn.CrossEntropyLoss()

        self.update(margin)

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

    def forward(self, input, label):
        # (batch, out_dim, k)
        cosine = torch.reshape(input, (-1, input.shape[1] // self.K, self.K))
        # (batch, out_dim)
        cosine, _ = torch.max(cosine, 2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)

        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = self.criterion(output, label)
        return loss


class TripletAngularMarginLoss(nn.Module):
    """A more robust triplet loss with hard positive/negative mining on angular margin instead of relative distance between d(a,p) and d(a,n).

    Args:
        margin (float, optional): angular margin. Defaults to 0.5.
        normalize_feature (bool, optional): whether to apply L2-norm in feature before computing distance(cos-similarity). Defaults to True.
        add_absolute (bool, optional): whether add absolute loss within d(a,p) or d(a,n). Defaults to False.
        absolute_loss_weight (float, optional): weight for absolute loss. Defaults to 1.0.
        ap_value (float, optional): weight for d(a, p). Defaults to 0.9.
        an_value (float, optional): weight for d(a, n). Defaults to 0.5.
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
        self.normalize_feature = normalize_feature
        self.add_absolute = add_absolute
        self.ap_value = ap_value
        self.an_value = an_value
        self.absolute_loss_weight = absolute_loss_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, target):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            target: ground truth labels with shape (batch_size)
        """
        if self.normalize_feature:
            inputs = F.normalize(inputs, p=2, dim=-1)

        loss_ce = self.criterion(inputs, target)
        bs = inputs.size(0)

        # compute distance(cos-similarity)
        dist = torch.matmul(inputs, inputs.t())

        # hard negative mining
        is_pos = target.expand(bs, bs).eq(target.expand(bs, bs).t())
        is_neg = target.expand(bs, bs).ne(target.expand(bs, bs).t())

        # `dist_ap` means distance(anchor, positive)
        dist_ap = dist[is_pos].view(bs, -1).min(dim=1, keepdim=True)[0]
        # `dist_an` means distance(anchor, negative)
        dist_an = dist[is_neg].view(bs, -1).max(dim=1, keepdim=True)[0]

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = F.relu(self.margin + dist_ap - dist_an).mean()

        if self.add_absolute:
            absolut_loss_ap = self.ap_value - dist_ap
            absolut_loss_ap = torch.where(absolut_loss_ap > 0, absolut_loss_ap, torch.zeros_like(absolut_loss_ap))

            absolut_loss_an = dist_an - self.an_value
            absolut_loss_an = torch.where(absolut_loss_an > 0, absolut_loss_an, torch.zeros_like(absolut_loss_an))

            loss = (absolut_loss_an.mean() + absolut_loss_ap.mean()) * self.absolute_loss_weight + loss
        loss = loss + loss_ce
        return loss

    def update(self, margin=0.5):
        self.margin = margin
