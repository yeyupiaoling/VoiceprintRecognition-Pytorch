import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from mvector.data_utils.audio import AudioSegment
from mvector.data_utils.featurizer import AudioFeaturizer
from mvector.utils.logger import setup_logger

logger = setup_logger(__name__)


class MVectorDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 audio_featurizer: AudioFeaturizer,
                 do_vad=True,
                 max_duration=3,
                 min_duration=0.5,
                 mode='train',
                 sample_rate=16000,
                 aug_conf={},
                 num_speakers=1000,
                 use_dB_normalization=True,
                 target_dB=-20):
        """音频数据加载器

        Args:
            data_list_path: 包含音频路径和标签的数据列表文件的路径
            audio_featurizer: 声纹特征提取器
            do_vad: 是否对音频进行语音活动检测（VAD）来裁剪静音部分
            max_duration: 最长的音频长度，大于这个长度会裁剪掉
            min_duration: 过滤最短的音频长度
            aug_conf: 用于指定音频增强的配置
            mode: 数据集模式。在训练模式下，数据集可能会进行一些数据增强的预处理
            sample_rate: 采样率
            num_speakers: 总说话人数量
            use_dB_normalization: 是否对音频进行音量归一化
            target_dB: 音量归一化的大小
        """
        super(MVectorDataset, self).__init__()
        assert mode in ['train', 'eval', 'create_data', 'extract_feature']
        self.do_vad = do_vad
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.mode = mode
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self.aug_conf = aug_conf
        self.num_speakers = num_speakers
        self.noises_path = None
        # 获取特征器
        self.audio_featurizer = audio_featurizer
        # 获取特征裁剪的大小
        self.max_feature_len = self.get_crop_feature_len()
        # 获取数据列表
        with open(data_list_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __getitem__(self, idx):
        # 分割数据文件路径和标签
        data_path, spk_id = self.lines[idx].replace('\n', '').split('\t')
        spk_id = int(spk_id)
        # 如果后缀名为.npy的文件，那么直接读取
        if data_path.endswith('.npy'):
            feature = np.load(data_path)
            if feature.shape[0] > self.max_feature_len:
                crop_start = random.randint(0, feature.shape[0] - self.max_feature_len) if self.mode == 'eval' else 0
                feature = feature[crop_start:crop_start + self.max_feature_len, :]
            feature = torch.tensor(feature, dtype=torch.float32)
        else:
            # 读取音频
            audio_segment = AudioSegment.from_file(data_path)
            # 裁剪静音
            if self.do_vad:
                audio_segment.vad()
            # 数据太短不利于训练
            if self.mode == 'train':
                if audio_segment.duration < self.min_duration:
                    return self.__getitem__(idx + 1 if idx < len(self.lines) - 1 else 0)
            # 重采样
            if audio_segment.sample_rate != self._target_sample_rate:
                audio_segment.resample(self._target_sample_rate)
            # 音频增强
            if self.mode == 'train':
                audio_segment, spk_id = self.augment_audio(audio_segment, spk_id, **self.aug_conf)
            # decibel normalization
            if self._use_dB_normalization:
                audio_segment.normalize(target_db=self._target_dB)
            # 裁剪需要的数据
            if self.mode != 'extract_feature' and audio_segment.duration > self.max_duration:
                audio_segment.crop(duration=self.max_duration, mode=self.mode)
            samples = torch.tensor(audio_segment.samples, dtype=torch.float32)
            feature = self.audio_featurizer(samples)
            feature = feature.squeeze(0)
        spk_id = torch.tensor(spk_id, dtype=torch.int64)
        return feature, spk_id

    def __len__(self):
        return len(self.lines)

    def get_crop_feature_len(self):
        samples = torch.randn((1, self.max_duration * self._target_sample_rate))
        feature = self.audio_featurizer(samples).squeeze(0)
        freq_len = feature.size(0)
        return freq_len

    # 音频增强
    def augment_audio(self,
                      audio_segment,
                      spk_id,
                      speed_perturb=False,
                      speed_perturb_3_class=False,
                      volume_perturb=False,
                      volume_aug_prob=0.2,
                      noise_dir=None,
                      noise_aug_prob=0.2):
        # 语速增强
        if speed_perturb:
            speeds = [1.0, 0.9, 1.1]
            speed_idx = random.randint(0, 2)
            speed_rate = speeds[speed_idx]
            if speed_rate != 1.0:
                audio_segment.change_speed(speed_rate)
            # 注意使用语速增强分类数量会大三倍
            if speed_perturb_3_class:
                spk_id = spk_id + self.num_speakers * speed_idx
        # 音量增强
        if volume_perturb and random.random() < volume_aug_prob:
            min_gain_dBFS, max_gain_dBFS = -15, 15
            gain = random.uniform(min_gain_dBFS, max_gain_dBFS)
            audio_segment.gain_db(gain)
        # 获取噪声文件
        if self.noises_path is None and noise_dir is not None:
            self.noises_path = []
            if noise_dir is not None and os.path.exists(noise_dir):
                for file in os.listdir(noise_dir):
                    self.noises_path.append(os.path.join(noise_dir, file))
        # 噪声增强
        if len(self.noises_path) > 0 and random.random() < noise_aug_prob:
            min_snr_dB, max_snr_dB = 10, 50
            # 随机选择一个noises_path中的一个
            noise_path = random.sample(self.noises_path, 1)[0]
            # 读取噪声音频
            noise_segment = AudioSegment.slice_from_file(noise_path)
            # 如果噪声采样率不等于audio_segment的采样率，则重采样
            if noise_segment.sample_rate != audio_segment.sample_rate:
                noise_segment.resample(audio_segment.sample_rate)
            # 随机生成snr_dB的值
            snr_dB = random.uniform(min_snr_dB, max_snr_dB)
            # 如果噪声的长度小于audio_segment的长度，则将噪声的前面的部分填充噪声末尾补长
            if noise_segment.duration < audio_segment.duration:
                diff_duration = audio_segment.num_samples - noise_segment.num_samples
                noise_segment._samples = np.pad(noise_segment.samples, (0, diff_duration), 'wrap')
            # 将噪声添加到audio_segment中，并将snr_dB调整到最小值和最大值之间
            audio_segment.add_noise(noise_segment, snr_dB)
        return audio_segment, spk_id
