import random

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm
from yeaudio.audio import AudioSegment
from yeaudio.augmentation import ReverbPerturbAugmentor, SpecAugmentor
from yeaudio.augmentation import SpeedPerturbAugmentor, VolumePerturbAugmentor, NoisePerturbAugmentor

from mvector.data_utils.featurizer import AudioFeaturizer


class MVectorDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 audio_featurizer: AudioFeaturizer,
                 max_duration=3,
                 min_duration=0.5,
                 mode='train',
                 sample_rate=16000,
                 aug_conf=None,
                 num_speakers=None,
                 use_dB_normalization=True,
                 target_dB=-20):
        """音频数据加载器

        Args:
            data_list_path: 包含音频路径和标签的数据列表文件的路径
            audio_featurizer: 声纹特征提取器
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
        assert mode in ['train', 'eval', 'extract_feature']
        self.data_list_path = data_list_path
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.mode = mode
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self.num_speakers = num_speakers
        self.speed_augment = None
        self.volume_augment = None
        self.noise_augment = None
        self.reverb_augment = None
        self.spec_augment = None
        # 获取特征器
        self.audio_featurizer = audio_featurizer
        # 获取特征裁剪的大小
        self.max_feature_len = self.get_crop_feature_len()
        # 获取数据列表
        with open(self.data_list_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        self.labels = [np.int64(line.strip().split('\t')[1]) for line in self.lines]
        if mode == 'train' and aug_conf is not None:
            # 获取数据增强器
            self.get_augmentor(aug_conf)
        # 评估模式下，数据列表需要排序
        if self.mode == 'eval':
            self.sort_list()

    def __getitem__(self, idx):
        # 分割数据文件路径和标签
        data_path, spk_id = self.lines[idx].replace('\n', '').split('\t')
        spk_id = int(spk_id)
        # 如果后缀名为.npy的文件，那么直接读取
        if data_path.endswith('.npy'):
            feature = np.load(data_path)
            if feature.shape[0] > self.max_feature_len:
                crop_start = random.randint(0, feature.shape[0] - self.max_feature_len) if self.mode == 'train' else 0
                feature = feature[crop_start:crop_start + self.max_feature_len, :]
            feature = torch.tensor(feature, dtype=torch.float32)
        else:
            # 读取音频
            audio_segment = AudioSegment.from_file(data_path)
            # 数据太短不利于训练
            if self.mode == 'train' or self.mode == 'extract_feature':
                if audio_segment.duration < self.min_duration:
                    return self.__getitem__(idx + 1 if idx < len(self.lines) - 1 else 0)
            # 重采样
            if audio_segment.sample_rate != self._target_sample_rate:
                audio_segment.resample(self._target_sample_rate)
            # 音频增强
            if self.mode == 'train':
                audio_segment, spk_id = self.augment_audio(audio_segment, spk_id)
            # decibel normalization
            if self._use_dB_normalization:
                audio_segment.normalize(target_db=self._target_dB)
            # 裁剪需要的数据
            if audio_segment.duration > self.max_duration:
                audio_segment.crop(duration=self.max_duration, mode=self.mode)
            samples = torch.tensor(audio_segment.samples, dtype=torch.float32)
            try:
                feature = self.audio_featurizer(samples)
            except Exception as e:
                logger.error(f"[{data_path}]特征提取失败，错误信息：{e}")
                return self.__getitem__(idx + 1 if idx < len(self.lines) - 1 else 0)
            feature = feature.squeeze(0)
        if self.mode == 'train' and self.spec_augment is not None:
            feature = self.spec_augment(feature.cpu().numpy())
            feature = torch.tensor(feature, dtype=torch.float32)
        spk_id = torch.tensor(spk_id, dtype=torch.int64)
        return feature, spk_id

    def __len__(self):
        return len(self.lines)

    # 获取特征裁剪的大小，对应max_duration音频提取特征后的长度
    def get_crop_feature_len(self):
        samples = torch.randn((1, int(self.max_duration * self._target_sample_rate)))
        feature = self.audio_featurizer(samples).squeeze(0)
        freq_len = feature.size(0)
        return freq_len

    # 数据列表需要排序
    def sort_list(self):
        lengths = []
        for line in tqdm(self.lines, desc=f"对列表[{self.data_list_path}]进行长度排序"):
            # 分割数据文件路径和标签
            data_path, _ = line.split('\t')
            if data_path.endswith('.npy'):
                feature = np.load(data_path)
                length = feature.shape[0]
                lengths.append(length)
            else:
                # 读取音频
                audio_segment = AudioSegment.from_file(data_path)
                length = audio_segment.duration
                lengths.append(length)
        # 对长度排序并获取索引
        sorted_indexes = np.argsort(lengths)
        self.lines = [self.lines[i] for i in sorted_indexes]

    # 获取数据增强器
    def get_augmentor(self, aug_conf):
        if aug_conf.speed is not None:
            self.speed_augment = SpeedPerturbAugmentor(num_speakers=self.num_speakers, **aug_conf.speed)
        if aug_conf.volume is not None:
            self.volume_augment = VolumePerturbAugmentor(**aug_conf.volume)
        if aug_conf.noise is not None:
            self.noise_augment = NoisePerturbAugmentor(**aug_conf.noise)
        if aug_conf.reverb is not None:
            self.reverb_augment = ReverbPerturbAugmentor(**aug_conf.reverb)
        if aug_conf.spec_aug is not None:
            self.spec_augment = SpecAugmentor(**aug_conf.spec_aug)

    # 音频增强
    def augment_audio(self, audio_segment, spk_id):
        if self.speed_augment is not None:
            audio_segment, spk_id = self.speed_augment(audio_segment, spk_id)
        if self.volume_augment is not None:
            audio_segment = self.volume_augment(audio_segment)
        if self.noise_augment is not None:
            audio_segment = self.noise_augment(audio_segment)
        if self.reverb_augment is not None:
            audio_segment = self.reverb_augment(audio_segment)
        return audio_segment, spk_id
