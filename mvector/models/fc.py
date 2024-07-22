import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerIdentification(nn.Module):
    def __init__(self,
                 input_dim,
                 num_speakers,
                 classifier_type='Cosine',
                 K=1,
                 num_blocks=0,
                 inter_dim=512):
        """The speaker identification model, which includes the speaker backbone network
           and the a linear transform to speaker class num in training

        Args:
            input_dim (nn.Module, class): embedding model output dim.
            num_speakers (_type_): the speaker class num in the training dataset
            classifier_type (str, optional): type of output layer to uses.
            K (int, optional): SubCenterLoss function parameter. It has to match the K of the classifier.
            num_blocks (int, optional): the linear layer transform between the embedding and the final linear layer. Defaults to 0.
            inter_dim (int, optional): the output dimension of dense layer. Defaults to 512.
        """
        super(SpeakerIdentification, self).__init__()
        self.classifier_type = classifier_type
        self.blocks = nn.ModuleList()

        for index in range(num_blocks):
            self.blocks.append(DenseLayer(input_dim, inter_dim, config_str='batchnorm'))
            input_dim = inter_dim

        if self.classifier_type == 'Cosine':
            self.weight = nn.Parameter(torch.FloatTensor(num_speakers * K, input_dim))
            nn.init.xavier_uniform_(self.weight)
        elif self.classifier_type == 'Linear':
            self.output = nn.Linear(input_dim, num_speakers)
        else:
            raise ValueError(f'不支持该输出层：{self.classifier_type}')

    def forward(self, features):
        # x: [B, dim]
        x = features
        for layer in self.blocks:
            x = layer(x)

        # normalized
        if self.classifier_type == 'Cosine':
            logits = F.linear(F.normalize(x), F.normalize(self.weight))
        else:
            logits = self.output(x)

        return {"features": features, "logits": logits}


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
