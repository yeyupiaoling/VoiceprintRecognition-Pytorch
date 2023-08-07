import torch
import torch.nn as nn


class SpecAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def freq_mask(self, x):
        batch, _, fea = x.shape
        mask_len = torch.randint(self.freq_mask_width[0], self.freq_mask_width[1], (batch, 1),
                                 device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, fea - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(fea, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)
        mask = mask.unsqueeze(1)
        x = x.masked_fill_(mask, 0.0)
        return x

    def time_mask(self, x):
        batch, time, _ = x.shape
        mask_len = torch.randint(self.time_mask_width[0], self.time_mask_width[1], (batch, 1),
                                 device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, time - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(time, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)
        mask = mask.unsqueeze(2)
        x = x.masked_fill_(mask, 0.0)
        return x

    def forward(self, x):
        x = self.freq_mask(x)
        x = self.time_mask(x)
        return x
