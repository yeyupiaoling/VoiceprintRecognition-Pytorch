import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class SpeakerIdentification(nn.Module):
    def __init__(self,
                 input_dim,
                 num_class,
                 loss_type='AAMLoss',
                 num_blocks=0,
                 inter_dim=512):
        """The speaker identification model, which includes the speaker backbone network
           and the a linear transform to speaker class num in training

        Args:
            input_dim (nn.Module, class): embedding model output dim.
            num_class (_type_): the speaker class num in the training dataset
            num_blocks (int, optional): the linear layer transform between the embedding and the final linear layer. Defaults to 0.
            inter_dim (int, optional): the output dimension of dense layer. Defaults to 512.
        """
        super(SpeakerIdentification, self).__init__()
        self.loss_type = loss_type
        self.blocks = nn.ModuleList()

        for index in range(num_blocks):
            self.blocks.append(DenseLayer(input_dim, inter_dim, config_str='batchnorm'))
            input_dim = inter_dim

        self.weight = nn.Parameter(torch.FloatTensor(num_class, input_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: [B, dim]
        for layer in self.blocks:
            x = layer(x)

        # normalized
        if self.loss_type == 'AAMLoss':
            logits = F.linear(F.normalize(x), F.normalize(self.weight))
        elif self.loss_type == 'AMLoss' or self.loss_type == 'ARMLoss':
            x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            x_norm = torch.div(x, x_norm)
            w_norm = torch.norm(self.weight, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            w_norm = torch.div(self.weight, w_norm)
            logits = torch.mm(x_norm, w_norm)
        else:
            logits = self.output(x)

        return logits


class DenseLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


class SpeakerIdentification1(nn.Module):
    def __init__(
            self,
            backbone,
            num_class=1,
            loss_type='AAMLoss',
            lin_blocks=0,
            lin_neurons=192,
            dropout=0.1, ):
        """The speaker identification model, which includes the speaker backbone network
           and the a linear transform to speaker class num in training

        Args:
            backbone (Paddle.nn.Layer class): the speaker identification backbone network model
            num_class (_type_): the speaker class num in the training dataset
            lin_blocks (int, optional): the linear layer transform between the embedding and the final linear layer. Defaults to 0.
            lin_neurons (int, optional): the output dimension of final linear layer. Defaults to 192.
            dropout (float, optional): the dropout factor on the embedding. Defaults to 0.1.
        """
        super(SpeakerIdentification1, self).__init__()
        # speaker idenfication backbone network model
        # the output of the backbond network is the target embedding
        self.backbone = backbone
        self.loss_type = loss_type
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # construct the speaker classifer
        input_size = self.backbone.emb_size
        self.blocks = list()
        for i in range(lin_blocks):
            self.blocks.extend([
                nn.BatchNorm1d(input_size),
                nn.Linear(in_features=input_size, out_features=lin_neurons),
            ])
            input_size = lin_neurons

        # the final layer
        if self.loss_type == 'AAMLoss':
            self.weight = Parameter(torch.FloatTensor(num_class, input_size), requires_grad=True)
            nn.init.xavier_normal_(self.weight, gain=1)
        elif self.loss_type == 'AMLoss' or self.loss_type == 'ARMLoss':
            self.weight = Parameter(torch.FloatTensor(input_size, num_class), requires_grad=True)
            nn.init.xavier_normal_(self.weight, gain=1)
        elif self.loss_type == 'CELoss':
            self.output = nn.Linear(input_size, num_class)
        else:
            raise Exception(f'没有{self.loss_type}损失函数！')

    def forward(self, x):
        """Do the speaker identification model forwrd,
           including the speaker embedding model and the classifier model network

        Args:
            x (paddle.Tensor): input audio feats,
                               shape=[batch, times, dimension]

        Returns:
            paddle.Tensor: return the logits of the feats
        """
        # x.shape: (N, L, C)
        x = self.backbone(x)  # (N, emb_size)
        if self.dropout is not None:
            x = self.dropout(x)

        for fc in self.blocks:
            x = fc(x)
        if self.loss_type == 'AAMLoss':
            logits = F.linear(F.normalize(x), F.normalize(self.weight, dim=-1))
        elif self.loss_type == 'AMLoss' or self.loss_type == 'ARMLoss':
            x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            x_norm = torch.div(x, x_norm)
            w_norm = torch.norm(self.weight, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            w_norm = torch.div(self.weight, w_norm)
            logits = torch.mm(x_norm, w_norm)
        else:
            logits = self.output(x)

        return logits
