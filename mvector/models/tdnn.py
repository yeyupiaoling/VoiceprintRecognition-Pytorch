import torch
import torch.nn as nn
import torch.nn.functional as F

from mvector.models.pooling import AttentiveStatsPool, TemporalAveragePooling
from mvector.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling


class TDNN(nn.Module):
    def __init__(self, input_size=80, channels=512, embd_dim=192, pooling_type="ASP"):
        super(TDNN, self).__init__()
        self.embd_dim = embd_dim
        self.td_layer1 = torch.nn.Conv1d(in_channels=input_size, out_channels=512, dilation=1, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.td_layer2 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=2, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.td_layer3 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=3, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.td_layer4 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=1, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.td_layer5 = torch.nn.Conv1d(in_channels=512, out_channels=channels, dilation=1, kernel_size=1, stride=1)

        if pooling_type == "ASP":
            self.pooling = AttentiveStatsPool(channels, 128)
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear = nn.Linear(channels * 2, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(channels, 128)
            self.bn5 = nn.BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
            self.bn5 = nn.BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear = nn.Linear(channels * 2, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

    def forward(self, x):
        """
        Compute embeddings.

        Args:
            x (torch.Tensor): Input data with shape (N, time, freq).

        Returns:
            torch.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        """
        x = x.transpose(2, 1)
        x = F.relu(self.td_layer1(x))
        x = self.bn1(x)
        x = F.relu(self.td_layer2(x))
        x = self.bn2(x)
        x = F.relu(self.td_layer3(x))
        x = self.bn3(x)
        x = F.relu(self.td_layer4(x))
        x = self.bn4(x)
        x = F.relu(self.td_layer5(x))
        out = self.bn5(self.pooling(x))
        out = self.bn6(self.linear(out))
        return out
