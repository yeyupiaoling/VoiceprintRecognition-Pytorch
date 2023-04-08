"""Contains the noise perturb augmentation model."""
import os
import random

import numpy as np

from mvector.data_utils.augmentor.base import AugmentorBase
from mvector.data_utils.audio import AudioSegment


class NoisePerturbAugmentor(AugmentorBase):
    """用于添加背景噪声的增强模型

    :param min_snr_dB: Minimal signal noise ratio, in decibels.
    :type min_snr_dB: float
    :param max_snr_dB: Maximal signal noise ratio, in decibels.
    :type max_snr_dB: float
    :param repetition: repetition noise sum
    :type repetition: int
    :param noise_dir: noise audio file dir.
    :type noise_dir: str
    """

    def __init__(self, min_snr_dB, max_snr_dB, repetition, noise_dir):
        self._min_snr_dB = min_snr_dB
        self._max_snr_dB = max_snr_dB
        self.repetition = repetition
        self.noises_path = []
        if os.path.exists(noise_dir):
            for file in os.listdir(noise_dir):
                self.noises_path.append(os.path.join(noise_dir, file))

    def transform_audio(self, audio_segment: AudioSegment):
        """Add background noise audio.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegmenet
        """
        if len(self.noises_path) > 0:
            for _ in range(random.randint(1, self.repetition)):
                # 随机选择一个noises_path中的一个
                noise_path = random.sample(self.noises_path, 1)[0]
                # 读取噪声音频
                noise_segment = AudioSegment.from_file(noise_path)
                # 如果噪声采样率不等于audio_segment的采样率，则重采样
                if noise_segment.sample_rate != audio_segment.sample_rate:
                    noise_segment.resample(audio_segment.sample_rate)
                # 随机生成snr_dB的值
                snr_dB = random.uniform(self._min_snr_dB, self._max_snr_dB)
                # 如果噪声的长度小于audio_segment的长度，则将噪声的前面的部分填充噪声末尾补长
                if noise_segment.duration < audio_segment.duration:
                    diff_duration = audio_segment.num_samples - noise_segment.num_samples
                    noise_segment._samples = np.pad(noise_segment.samples, (0, diff_duration), 'wrap')
                # 将噪声添加到audio_segment中，并将snr_dB调整到最小值和最大值之间
                audio_segment.add_noise(noise_segment, snr_dB)
