import random

import numpy as np


class SpecSubAugmentor(object):
    """Do spec substitute. Inplace operation

    https://arxiv.org/abs/2106.05642
    """

    def __init__(self,
                 max_t=20,
                 num_t_sub=3):
        """SpecAugmentor class.
        Args:
            :param max_t: 时间替换的最大宽度
            :type max_t: int
            :param num_t_sub: 申请替换的时间数
            :type num_t_sub: int
        """
        super().__init__()
        self.max_t = max_t
        self.num_t_sub = num_t_sub

    def __call__(self, x, train=True):
        if not train:
            return x
        return self.transform_feature(x)

    def transform_feature(self, x: np.ndarray):
        y = x.copy()
        max_frames = y.shape[0]
        for i in range(self.num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, self.max_t)
            end = min(max_frames, start + length)
            pos = random.randint(0, start)
            y[start:end, :] = x[start - pos:end - pos, :]
        return y
