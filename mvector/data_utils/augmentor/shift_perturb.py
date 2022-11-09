"""Contains the volume perturb augmentation model."""
import random

from mvector.data_utils.audio import AudioSegment

from mvector.data_utils.augmentor.base import AugmentorBase


class ShiftPerturbAugmentor(AugmentorBase):
    """添加随机位移扰动的增强模型
    
    :param min_shift_ms: Minimal shift in milliseconds.
    :type min_shift_ms: float
    :param max_shift_ms: Maximal shift in milliseconds.
    :type max_shift_ms: float
    """

    def __init__(self, min_shift_ms, max_shift_ms):
        self._min_shift_ms = min_shift_ms
        self._max_shift_ms = max_shift_ms

    def transform_audio(self, audio_segment: AudioSegment):
        """Shift audio.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        shift_ms = random.uniform(self._min_shift_ms, self._max_shift_ms)
        audio_segment.shift(shift_ms)
