import random

import numpy as np
from PIL import Image
from PIL.Image import BICUBIC


class SpecAugmentor(object):
    """Augmentation model for Time warping, Frequency masking, Time masking.

    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
        https://arxiv.org/abs/1904.08779
    SpecAugment on Large Scale Datasets
        https://arxiv.org/abs/1912.05533
    """

    def __init__(self,
                 max_f_ratio=0.15,
                 n_freq_masks=2,
                 max_t_ratio=0.05,
                 n_time_masks=2,
                 inplace=True,
                 max_time_warp=5,
                 replace_with_zero=False):
        """SpecAugment class.
        Args:
            :param max_t_ratio: 时间屏蔽的比例
            :type max_t_ratio: float
            :param n_freq_masks: 频率屏蔽数量
            :type n_freq_masks: int
            :param max_f_ratio: 频率屏蔽的比例
            :type max_f_ratio: float
            :param n_time_masks: 时间屏蔽数量
            :type n_time_masks: int
            :param inplace: 用结果覆盖
            :type inplace: bool
            :param replace_with_zero: 如果真的话，在pad补0，否则使用平均值
            :type replace_with_zero: bool
        """
        super().__init__()
        self.inplace = inplace
        self.replace_with_zero = replace_with_zero
        self.max_time_warp = max_time_warp
        self.max_t_ratio = max_t_ratio
        self.max_f_ratio = max_f_ratio
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def time_warp(self, x):
        """time warp for spec augment
        move random center frame by the random width ~ uniform(-window, window)

        Args:
            x (np.ndarray): spectrogram (time, freq)
            mode (str): PIL or sparse_image_warp

        Raises:
            NotImplementedError: [description]
            NotImplementedError: [description]

        Returns:
            np.ndarray: time warped spectrogram (time, freq)
        """
        window = self.max_time_warp
        if window == 0:
            return x

        t = x.shape[0]
        if t - window <= window:
            return x
        # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
        center = random.randrange(window, t - window)
        warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1
        left = Image.fromarray(x[:center]).resize((x.shape[1], warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped), BICUBIC)
        if self.inplace:
            x[:warped] = left
            x[warped:] = right
            return x
        return np.concatenate((left, right), 0)

    def freq_mask(self, x, replace_with_zero=False):
        """freq mask

        Args:
            x (np.ndarray): spectrogram (time, freq)
            replace_with_zero (bool, optional): Defaults to False.

        Returns:
            np.ndarray: freq mask spectrogram (time, freq)
        """
        cloned = x if self.inplace else x.copy()
        max_freq = cloned.shape[1]
        max_f = int(max_freq * self.max_f_ratio)
        for i in range(self.n_freq_masks):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            if replace_with_zero:
                cloned[:, start:end] = 0
            else:
                cloned[:, start:end] = cloned.mean()
        return cloned

    def mask_time(self, x, replace_with_zero=False):
        """time mask

        Args:
            x (np.ndarray): spectrogram (time, freq)
            replace_with_zero (bool, optional): Defaults to False.

        Returns:
            np.ndarray: time mask spectrogram (time, freq)
        """
        cloned = x if self.inplace else x.copy()
        max_frames = cloned.shape[0]
        max_t = int(max_frames * self.max_t_ratio)
        for i in range(self.n_time_masks):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            if replace_with_zero:
                cloned[start:end, :] = 0
            else:
                cloned[start:end, :] = cloned.mean()
        return cloned

    def __call__(self, x, train=True):
        if not train:
            return x
        return self.transform_feature(x)

    def transform_feature(self, x: np.ndarray):
        """
        Args:
            x (np.ndarray): `[T, F]`
        Returns:
            x (np.ndarray): `[T, F]`
        """
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2
        x = self.time_warp(x)
        x = self.freq_mask(x, self.replace_with_zero)
        x = self.mask_time(x, self.replace_with_zero)
        return x
