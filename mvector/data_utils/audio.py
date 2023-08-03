import copy
import io
import os
import random

import numpy as np
import resampy
import soundfile

from mvector.data_utils.utils import buf_to_float, vad, decode_audio


class AudioSegment(object):
    """Monaural audio segment abstraction.

    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate):
        """Create audio segment from samples.

        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        self._samples = self._convert_samples_to_float32(samples)
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    def __eq__(self, other):
        """返回两个对象是否相等"""
        if type(other) is not type(self):
            return False
        if self._sample_rate != other._sample_rate:
            return False
        if self._samples.shape != other._samples.shape:
            return False
        if np.any(self.samples != other._samples):
            return False
        return True

    def __ne__(self, other):
        """返回两个对象是否不相等"""
        return not self.__eq__(other)

    def __str__(self):
        """返回该音频的信息"""
        return ("%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, "
                "rms=%.2fdB" % (type(self), self.num_samples, self.sample_rate, self.duration, self.rms_db))

    @classmethod
    def from_file(cls, file):
        """从音频文件创建音频段

        :param file: 文件路径，或者文件对象
        :type file: str, BufferedReader
        :return: 音频片段实例
        :rtype: AudioSegment
        """
        assert os.path.exists(file), f'文件不存在，请检查路径：{file}'
        try:
            samples, sample_rate = soundfile.read(file, dtype='float32')
        except:
            # 支持更多格式数据
            sample_rate = 16000
            samples = decode_audio(file=file, sample_rate=sample_rate)
        return cls(samples, sample_rate)

    @classmethod
    def slice_from_file(cls, file, start=None, end=None):
        """只加载一小段音频，而不需要将整个文件加载到内存中，这是非常浪费的。

        :param file: 输入音频文件路径或文件对象
        :type file: str|file
        :param start: 开始时间，单位为秒。如果start是负的，则它从末尾开始计算。如果没有提供，这个函数将从最开始读取。
        :type start: float
        :param end: 结束时间，单位为秒。如果end是负的，则它从末尾开始计算。如果没有提供，默认的行为是读取到文件的末尾。
        :type end: float
        :return: AudioSegment输入音频文件的指定片的实例。
        :rtype: AudioSegment
        :raise ValueError: 如开始或结束的设定不正确，例如时间不允许。
        """
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = 0. if start is None else round(start, 3)
        end = duration if end is None else round(end, 3)
        # 从末尾开始计
        if start < 0.0: start += duration
        if end < 0.0: end += duration
        # 保证数据不越界
        if start < 0.0: start = 0.0
        if end > duration: end = duration
        if end < 0.0:
            raise ValueError("切片结束位置(%f s)越界" % end)
        if start > end:
            raise ValueError("切片开始位置(%f s)晚于切片结束位置(%f s)" % (start, end))
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        data = sndfile.read(frames=end_frame - start_frame, dtype='float32')
        return cls(data, sample_rate)

    @classmethod
    def from_bytes(cls, data):
        """从包含音频样本的字节创建音频段

        :param data: 包含音频样本的字节
        :type data: bytes
        :return: 音频部分实例
        :rtype: AudioSegment
        """
        samples, sample_rate = soundfile.read(io.BytesIO(data), dtype='float32')
        return cls(samples, sample_rate)

    @classmethod
    def from_pcm_bytes(cls, data, channels=1, samp_width=2, sample_rate=16000):
        """从包含无格式PCM音频的字节创建音频

        :param data: 包含音频样本的字节
        :type data: bytes
        :param channels: 音频的通道数
        :type channels: int
        :param samp_width: 音频采样的宽度，如np.int16为2
        :type samp_width: int
        :param sample_rate: 音频样本采样率
        :type sample_rate: int
        :return: 音频部分实例
        :rtype: AudioSegment
        """
        samples = buf_to_float(data, n_bytes=samp_width)
        if channels > 1:
            samples = samples.reshape(-1, channels)
        return cls(samples, sample_rate)

    @classmethod
    def from_ndarray(cls, data, sample_rate=16000):
        """从numpy.ndarray创建音频段

        :param data: numpy.ndarray类型的音频数据
        :type data: ndarray
        :param sample_rate: 音频样本采样率
        :type sample_rate: int
        :return: 音频部分实例
        :rtype: AudioSegment
        """
        return cls(data, sample_rate)

    @classmethod
    def concatenate(cls, *segments):
        """将任意数量的音频片段连接在一起

        :param *segments: 输入音频片段被连接
        :type *segments: tuple of AudioSegment
        :return: Audio segment instance as concatenating results.
        :rtype: AudioSegment
        :raises ValueError: If the number of segments is zero, or if the 
                            sample_rate of any segments does not match.
        :raises TypeError: If any segment is not AudioSegment instance.
        """
        # Perform basic sanity-checks.
        if len(segments) == 0:
            raise ValueError("没有音频片段被给予连接")
        sample_rate = segments[0]._sample_rate
        for seg in segments:
            if sample_rate != seg._sample_rate:
                raise ValueError("能用不同的采样率连接片段")
            if type(seg) is not cls:
                raise TypeError("只有相同类型的音频片段可以连接")
        samples = np.concatenate([seg.samples for seg in segments])
        return cls(samples, sample_rate)

    @classmethod
    def make_silence(cls, duration, sample_rate):
        """创建给定持续时间和采样率的静音音频段

        :param duration: 静音的时间，以秒为单位
        :type duration: float
        :param sample_rate: 音频采样率
        :type sample_rate: float
        :return: 给定持续时间的静音AudioSegment实例
        :rtype: AudioSegment
        """
        samples = np.zeros(int(duration * sample_rate))
        return cls(samples, sample_rate)

    def to_wav_file(self, filepath, dtype='float32'):
        """保存音频段到磁盘为wav文件
        
        :param filepath: WAV文件路径或文件对象，以保存音频段
        :type filepath: str|file
        :param dtype: Subtype for audio file. Options: 'int16', 'int32',
                      'float32', 'float64'. Default is 'float32'.
        :type dtype: str
        :raises TypeError: If dtype is not supported.
        """
        samples = self._convert_samples_from_float32(self._samples, dtype)
        subtype_map = {
            'int16': 'PCM_16',
            'int32': 'PCM_32',
            'float32': 'FLOAT',
            'float64': 'DOUBLE'
        }
        soundfile.write(
            filepath,
            samples,
            self._sample_rate,
            format='WAV',
            subtype=subtype_map[dtype])

    def superimpose(self, other):
        """将另一个段的样本添加到这个段的样本中(以样本方式添加，而不是段连接)。

        :param other: 包含样品的片段被添加进去
        :type other: AudioSegments
        :raise TypeError: 如果两个片段的类型不匹配
        :raise ValueError: 不能添加不同类型的段
        """
        if not isinstance(other, type(self)):
            raise TypeError("不能添加不同类型的段: %s 和 %s" % (type(self), type(other)))
        if self._sample_rate != other._sample_rate:
            raise ValueError("采样率必须匹配才能添加片段")
        if len(self._samples) != len(other._samples):
            raise ValueError("段长度必须匹配才能添加段")
        self._samples += other._samples

    def to_bytes(self, dtype='float32'):
        """创建包含音频内容的字节字符串
        
        :param dtype: Data type for export samples. Options: 'int16', 'int32',
                      'float32', 'float64'. Default is 'float32'.
        :type dtype: str
        :return: Byte string containing audio content.
        :rtype: str
        """
        samples = self._convert_samples_from_float32(self._samples, dtype)
        return samples.tostring()

    def to(self, dtype='int16'):
        """类型转换

        :param dtype: Data type for export samples. Options: 'int16', 'int32',
                      'float32', 'float64'. Default is 'float32'.
        :type dtype: str
        :return: np.ndarray containing `dtype` audio content.
        :rtype: str
        """
        samples = self._convert_samples_from_float32(self._samples, dtype)
        return samples

    def gain_db(self, gain):
        """对音频施加分贝增益。

        Note that this is an in-place transformation.
        
        :param gain: Gain in decibels to apply to samples. 
        :type gain: float|1darray
        """
        self._samples *= 10.**(gain / 20.)

    def change_speed(self, speed_rate):
        """通过线性插值改变音频速度

        :param speed_rate: Rate of speed change:
                           speed_rate > 1.0, speed up the audio;
                           speed_rate = 1.0, unchanged;
                           speed_rate < 1.0, slow down the audio;
                           speed_rate <= 0.0, not allowed, raise ValueError.
        :type speed_rate: float
        :raises ValueError: If speed_rate <= 0.0.
        """
        if speed_rate == 1.0:
            return
        if speed_rate <= 0:
            raise ValueError("速度速率应大于零")
        old_length = self._samples.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        self._samples = np.interp(new_indices, old_indices, self._samples).astype(np.float32)

    def normalize(self, target_db=-20, max_gain_db=300.0):
        """将音频归一化，使其具有所需的有效值(以分贝为单位)

        :param target_db: Target RMS value in decibels. This value should be
                          less than 0.0 as 0.0 is full-scale audio.
        :type target_db: float
        :param max_gain_db: Max amount of gain in dB that can be applied for
                            normalization. This is to prevent nans when
                            attempting to normalize a signal consisting of
                            all zeros.
        :type max_gain_db: float
        :raises ValueError: If the required gain to normalize the segment to
                            the target_db value exceeds max_gain_db.
        """
        if -np.inf == self.rms_db: return
        gain = target_db - self.rms_db
        if gain > max_gain_db:
            raise ValueError(
                "无法将段规范化到 %f dB，因为可能的增益已经超过max_gain_db (%f dB)" % (target_db, max_gain_db))
        self.gain_db(min(max_gain_db, target_db - self.rms_db))

    def resample(self, target_sample_rate, filter='kaiser_best'):
        """按目标采样率重新采样音频

        Note that this is an in-place transformation.

        :param target_sample_rate: Target sample rate.
        :type target_sample_rate: int
        :param filter: The resampling filter to use one of {'kaiser_best', 'kaiser_fast'}.
        :type filter: str
        """
        self._samples = resampy.resample(self.samples, self.sample_rate, target_sample_rate, filter=filter)
        self._sample_rate = target_sample_rate

    def pad_silence(self, duration, sides='both'):
        """在这个音频样本上加一段静音

        Note that this is an in-place transformation.

        :param duration: Length of silence in seconds to pad.
        :type duration: float
        :param sides: Position for padding:
                     'beginning' - adds silence in the beginning;
                     'end' - adds silence in the end;
                     'both' - adds silence in both the beginning and the end.
        :type sides: str
        :raises ValueError: If sides is not supported.
        """
        if duration == 0.0:
            return self
        cls = type(self)
        silence = self.make_silence(duration, self._sample_rate)
        if sides == "beginning":
            padded = cls.concatenate(silence, self)
        elif sides == "end":
            padded = cls.concatenate(self, silence)
        elif sides == "both":
            padded = cls.concatenate(silence, self, silence)
        else:
            raise ValueError("Unknown value for the sides %s" % sides)
        self._samples = padded._samples

    def shift(self, shift_ms):
        """音频偏移。如果shift_ms为正，则随时间提前移位;如果为负，则随时间延迟移位。填补静音以保持持续时间不变。

        Note that this is an in-place transformation.

        :param shift_ms: Shift time in millseconds. If positive, shift with
                         time advance; if negative; shift with time delay.
        :type shift_ms: float
        :raises ValueError: If shift_ms is longer than audio duration.
        """
        if abs(shift_ms) / 1000.0 > self.duration:
            raise ValueError("shift_ms的绝对值应该小于音频持续时间")
        shift_samples = int(shift_ms * self._sample_rate / 1000)
        if shift_samples > 0:
            # time advance
            self._samples[:-shift_samples] = self._samples[shift_samples:]
            self._samples[-shift_samples:] = 0
        elif shift_samples < 0:
            # time delay
            self._samples[-shift_samples:] = self._samples[:shift_samples]
            self._samples[:-shift_samples] = 0

    def subsegment(self, start_sec=None, end_sec=None):
        """在给定的边界之间切割音频片段

        Note that this is an in-place transformation.

        :param start_sec: Beginning of subsegment in seconds.
        :type start_sec: float
        :param end_sec: End of subsegment in seconds.
        :type end_sec: float
        :raise ValueError: If start_sec or end_sec is incorrectly set, e.g. out
                           of bounds in time.
        """
        start_sec = 0.0 if start_sec is None else start_sec
        end_sec = self.duration if end_sec is None else end_sec
        if start_sec < 0.0:
            start_sec = self.duration + start_sec
        if end_sec < 0.0:
            end_sec = self.duration + end_sec
        if start_sec < 0.0:
            raise ValueError("切片起始位置(%f s)越界" % start_sec)
        if end_sec < 0.0:
            raise ValueError("切片结束位置(%f s)越界" % end_sec)
        if start_sec > end_sec:
            raise ValueError("切片的起始位置(%f s)晚于结束位置(%f s)" % (start_sec, end_sec))
        if end_sec > self.duration:
            raise ValueError("切片结束位置(%f s)越界(> %f s)" % (end_sec, self.duration))
        start_sample = int(round(start_sec * self._sample_rate))
        end_sample = int(round(end_sec * self._sample_rate))
        self._samples = self._samples[start_sample:end_sample]

    def random_subsegment(self, subsegment_length):
        """随机剪切指定长度的音频片段

        Note that this is an in-place transformation.

        :param subsegment_length: Subsegment length in seconds.
        :type subsegment_length: float
        :raises ValueError: If the length of subsegment is greater than
                            the origineal segemnt.
        """
        if subsegment_length > self.duration:
            raise ValueError("Length of subsegment must not be greater "
                             "than original segment.")
        start_time = random.uniform(0.0, self.duration - subsegment_length)
        self.subsegment(start_time, start_time + subsegment_length)

    def add_noise(self,
                  noise,
                  snr_dB,
                  max_gain_db=300.0):
        """以特定的信噪比添加给定的噪声段。如果噪声段比该噪声段长，则从该噪声段中采样匹配长度的随机子段。

        Note that this is an in-place transformation.

        :param noise: Noise signal to add.
        :type noise: AudioSegment
        :param snr_dB: Signal-to-Noise Ratio, in decibels.
        :type snr_dB: float
        :param max_gain_db: Maximum amount of gain to apply to noise signal
                            before adding it in. This is to prevent attempting
                            to apply infinite gain to a zero signal.
        :type max_gain_db: float
        :raises ValueError: If the sample rate does not match between the two
                            audio segments, or if the duration of noise segments
                            is shorter than original audio segments.
        """
        if noise.sample_rate != self.sample_rate:
            raise ValueError("噪声采样率(%d Hz)不等于基信号采样率(%d Hz)" % (noise.sample_rate, self.sample_rate))
        if noise.duration < self.duration:
            raise ValueError("噪声信号(%f秒)必须至少与基信号(%f秒)一样长" % (noise.duration, self.duration))
        noise_gain_db = min(self.rms_db - noise.rms_db - snr_dB, max_gain_db)
        noise_new = copy.deepcopy(noise)
        noise_new.random_subsegment(self.duration)
        noise_new.gain_db(noise_gain_db)
        self.superimpose(noise_new)

    def vad(self, top_db=20, overlap=200):
        self._samples = vad(wav=self._samples, top_db=top_db, overlap=overlap)

    # 裁剪音频
    def crop(self, duration, mode='eval'):
        if self.duration > duration:
            if mode == 'train':
                self.random_subsegment(duration)
            else:
                self.subsegment(end_sec=duration)

    @property
    def samples(self):
        """返回音频样本

        :return: Audio samples.
        :rtype: ndarray
        """
        return self._samples.copy()

    @property
    def sample_rate(self):
        """返回音频采样率

        :return: Audio sample rate.
        :rtype: int
        """
        return self._sample_rate

    @property
    def num_samples(self):
        """返回样品数量

        :return: Number of samples.
        :rtype: int
        """
        return self._samples.shape[0]

    @property
    def duration(self):
        """返回音频持续时间

        :return: Audio duration in seconds.
        :rtype: float
        """
        return self._samples.shape[0] / float(self._sample_rate)

    @property
    def rms_db(self):
        """返回以分贝为单位的音频均方根能量

        :return: Root mean square energy in decibels.
        :rtype: float
        """
        # square root => multiply by 10 instead of 20 for dBs
        mean_square = np.mean(self._samples ** 2)
        return 10 * np.log10(mean_square)

    def _convert_samples_to_float32(self, samples):
        """Convert sample type to float32.

        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= (1. / 2 ** (bits - 1))
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    def _convert_samples_from_float32(self, samples, dtype):
        """Convert sample type from float32 to dtype.

        Audio sample type is usually integer or float-point. For integer
        type, float32 will be rescaled from [-1, 1] to the maximum range
        supported by the integer type.

        This is for writing a audio file.
        """
        dtype = np.dtype(dtype)
        output_samples = samples.copy()
        if dtype in np.sctypes['int']:
            bits = np.iinfo(dtype).bits
            output_samples *= (2 ** (bits - 1) / 1.)
            min_val = np.iinfo(dtype).min
            max_val = np.iinfo(dtype).max
            output_samples[output_samples > max_val] = max_val
            output_samples[output_samples < min_val] = min_val
        elif samples.dtype in np.sctypes['float']:
            min_val = np.finfo(dtype).min
            max_val = np.finfo(dtype).max
            output_samples[output_samples > max_val] = max_val
            output_samples[output_samples < min_val] = min_val
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return output_samples.astype(dtype)
