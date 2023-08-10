import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mvector.models.pooling import TemporalAveragePooling, TemporalStatsPool


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'


def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution without padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels * 2, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x, ds_y):
        xa = torch.cat((x, ds_y), dim=1)
        x_att = self.local_att(xa)
        x_att = 1.0 + torch.tanh(x_att)
        xo = torch.mul(x, x_att) + torch.mul(ds_y, 2.0 - x_att)

        return xo


class BasicBlockERes2Net(nn.Module):

    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
        super(BasicBlockERes2Net, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class BasicBlockERes2Net_diff_AFF(nn.Module):

    def __init__(self, expansion, in_planes, planes, stride=1, base_width=32, scale=2):
        super(BasicBlockERes2Net_diff_AFF, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.nums = scale

        convs = []
        fuse_models = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        for j in range(self.nums - 1):
            fuse_models.append(AFF(channels=width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = self.fuse_models[i - 1](sp, spx[i])

            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class ERes2Net(nn.Module):
    def __init__(self,
                 input_size,
                 block=BasicBlockERes2Net,
                 block_fuse=BasicBlockERes2Net_diff_AFF,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=32,
                 mul_channel=1,
                 expansion=2,
                 base_width=32,
                 scale=2,
                 embd_dim=192,
                 pooling_type='TSTP',
                 two_emb_layer=False):
        super(ERes2Net, self).__init__()
        self.in_planes = m_channels
        self.expansion = expansion
        self.feat_dim = input_size
        self.embd_dim = embd_dim
        self.stats_dim = int(input_size / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1,
                                       base_width=base_width, scale=scale)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2,
                                       base_width=base_width, scale=scale)
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2,
                                       base_width=base_width, scale=scale)
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2,
                                       base_width=base_width, scale=scale)

        # Downsampling module for each layer
        self.layer1_downsample = nn.Conv2d(m_channels * 2 * mul_channel, m_channels * 4 * mul_channel, kernel_size=3,
                                           padding=1, stride=2, bias=False)
        self.layer2_downsample = nn.Conv2d(m_channels * 4 * mul_channel, m_channels * 8 * mul_channel, kernel_size=3,
                                           padding=1, stride=2, bias=False)
        self.layer3_downsample = nn.Conv2d(m_channels * 8 * mul_channel, m_channels * 16 * mul_channel, kernel_size=3,
                                           padding=1, stride=2, bias=False)
        self.fuse_mode12 = AFF(channels=m_channels * 4 * mul_channel)
        self.fuse_mode123 = AFF(channels=m_channels * 8 * mul_channel)
        self.fuse_mode1234 = AFF(channels=m_channels * 16 * mul_channel)

        self.n_stats = 1 if pooling_type == 'TAP' else 2
        if pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
        elif pooling_type == "TSTP":
            self.pooling = TemporalStatsPool()
        else:
            raise Exception(f'没有{pooling_type}池化层！')

        self.seg_1 = nn.Linear(self.stats_dim * self.expansion * self.n_stats, embd_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embd_dim, affine=False)
            self.seg_2 = nn.Linear(embd_dim, embd_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride, base_width, scale):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.expansion, self.in_planes, planes, stride, base_width, scale))
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out1_downsample = self.layer1_downsample(out1)
        fuse_out12 = self.fuse_mode12(out2, out1_downsample)
        out3 = self.layer3(out2)
        fuse_out12_downsample = self.layer2_downsample(fuse_out12)
        fuse_out123 = self.fuse_mode123(out3, fuse_out12_downsample)
        out4 = self.layer4(out3)
        fuse_out123_downsample = self.layer3_downsample(fuse_out123)
        fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_downsample)
        stats = self.pooling(fuse_out1234)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a
