import torch
import torch.nn as nn

from mvector.models.pooling import AttentiveStatisticsPooling, TemporalAveragePooling
from mvector.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling
from mvector.models.utils import Conv1d, length_to_mask, TDNNBlock, BatchNorm1d


class Res2NetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        """Implementation of Res2Net Block with dilation
           The paper is refered as "Res2Net: A New Multi-scale Backbone Architecture",
           whose url is https://arxiv.org/abs/1904.01169
        Args:
            in_channels (int): input channels or input dimensions
            out_channels (int): output channels or output dimensions
            scale (int, optional): scale in res2net bolck. Defaults to 8.
            dilation (int, optional): dilation of 1-d convolution in TDNN block. Defaults to 1.
        """
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x):
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = torch.cat(y, dim=1)
        return y


class SEBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        """Implementation of SEBlock
           The paper is refered as "Squeeze-and-Excitation Networks"
           whose url is https://arxiv.org/abs/1709.01507
        Args:
            in_channels (int): input channels or input data dimensions
            se_channels (_type_): _description_
            out_channels (int): output channels or output data dimensions
        """
        super(SEBlock, self).__init__()

        self.conv1 = Conv1d(in_channels=in_channels, out_channels=se_channels, kernel_size=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = Conv1d(in_channels=se_channels, out_channels=out_channels, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, lengths=None):
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        return s * x


class SERes2NetBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            res2net_scale=8,
            se_channels=128,
            kernel_size=1,
            dilation=1,
            activation=torch.nn.ReLU,
            groups=1, ):
        """Implementation of Squeeze-Extraction Res2Blocks in ECAPA-TDNN network model
           The paper is refered "Squeeze-and-Excitation Networks"
           whose url is: https://arxiv.org/pdf/1709.01507.pdf
        Args:
            in_channels (int): input channels or input data dimensions
            out_channels (int): output channels or output data dimensions
            res2net_scale (int, optional): scale in the res2net block. Defaults to 8.
            se_channels (int, optional): embedding dimensions of res2net block. Defaults to 128.
            kernel_size (int, optional): kernel size of 1-d convolution in TDNN block. Defaults to 1.
            dilation (int, optional): dilation of 1-d convolution in TDNN block. Defaults to 1.
            activation (paddle.nn.class, optional): activation function. Defaults to nn.ReLU.
        """
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(in_channels,
                               out_channels,
                               kernel_size=1,
                               dilation=1,
                               activation=activation,
                               groups=groups, )
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TDNNBlock(out_channels,
                               out_channels,
                               kernel_size=1,
                               dilation=1,
                               activation=activation,
                               groups=groups, )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1, )

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual


class EcapaTdnn(torch.nn.Module):
    def __init__(
            self,
            input_size,
            embd_dim=192,
            pooling_type="ASP",
            activation=nn.ReLU,
            channels=[512, 512, 512, 512, 1536],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            res2net_scale=8,
            se_channels=128,
            global_context=True,
            groups=[1, 1, 1, 1, 1], ):
        """Implementation of ECAPA-TDNN backbone model network
           The paper is refered as "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification"
           whose url is: https://arxiv.org/abs/2005.07143
        Args:
            input_size (_type_): input fature dimension
            embd_dim (int, optional): speaker embedding size. Defaults to 192.
            activation (paddle.nn.class, optional): activation function. Defaults to nn.ReLU.
            channels (list, optional): inter embedding dimension. Defaults to [512, 512, 512, 512, 1536].
            kernel_sizes (list, optional): kernel size of 1-d convolution in TDNN block . Defaults to [5, 3, 3, 3, 1].
            dilations (list, optional): dilations of 1-d convolution in TDNN block. Defaults to [1, 2, 3, 4, 1].
            attention_channels (int, optional): attention dimensions. Defaults to 128.
            res2net_scale (int, optional): scale value in res2net. Defaults to 8.
            se_channels (int, optional): dimensions of squeeze-excitation block. Defaults to 128.
            global_context (bool, optional): global context flag. Defaults to True.
        """
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
                groups[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(channels[-1],
                             channels[-1],
                             kernel_sizes[-1],
                             dilations[-1],
                             activation,
                             groups=groups[-1], )

        # Attentive Statistical Pooling
        cat_channels = channels[-1]
        self.embd_dim = embd_dim
        if pooling_type == "ASP":
            self.asp = AttentiveStatisticsPooling(channels[-1],
                                                  attention_channels=attention_channels,
                                                  global_context=global_context)
            self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)
            # Final linear transformation
            self.fc = Conv1d(in_channels=channels[-1] * 2,
                             out_channels=self.embd_dim,
                             kernel_size=1)
        elif pooling_type == "SAP":
            self.asp = SelfAttentivePooling(cat_channels, 128)
            self.asp_bn = nn.BatchNorm1d(cat_channels)
            # Final linear transformation
            self.fc = Conv1d(in_channels=cat_channels,
                             out_channels=self.embd_dim,
                             kernel_size=1)
        elif pooling_type == "TAP":
            self.asp = TemporalAveragePooling()
            self.asp_bn = nn.BatchNorm1d(cat_channels)
            # Final linear transformation
            self.fc = Conv1d(in_channels=cat_channels,
                             out_channels=self.embd_dim,
                             kernel_size=1)
        elif pooling_type == "TSP":
            self.asp = TemporalStatisticsPooling()
            self.asp_bn = nn.BatchNorm1d(cat_channels * 2)
            # Final linear transformation
            self.fc = Conv1d(in_channels=cat_channels * 2,
                             out_channels=self.embd_dim,
                             kernel_size=1)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

    def forward(self, x, lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)

        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x)
        x = self.asp_bn(x)
        x = x.unsqueeze(2)
        # Final linear transformation
        x = self.fc(x).squeeze(-1)  # (N, emb_size, 1) -> (N, emb_size)

        return x
