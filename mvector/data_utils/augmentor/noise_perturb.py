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
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        if len(self.noises_path) > 0:
            for _ in range(random.randint(1, self.repetition)):
                noise_path = random.sample(self.noises_path, 1)[0]
                noise_segment = AudioSegment.from_file(noise_path)
                snr_dB = random.uniform(self._min_snr_dB, self._max_snr_dB)
                if noise_segment.samples.shape[0] < audio_segment.samples.shape[0]:
                    diff_duration = audio_segment.samples.shape[0] - noise_segment.samples.shape[0]
                    noise_segment._samples = np.pad(noise_segment.samples, (0, diff_duration), 'wrap')
                audio_segment.add_noise(noise_segment, snr_dB, allow_downsampling=True)
