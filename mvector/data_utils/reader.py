import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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
        assert mode in ['train', 'eval', 'extract_feature']
        self.data_list_path = data_list_path
        self.do_vad = do_vad
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.mode = mode
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self.aug_conf = aug_conf
        self.num_speakers = num_speakers
        # 获取特征器
        self.audio_featurizer = audio_featurizer
        # 获取特征裁剪的大小
        self.max_feature_len = self.get_crop_feature_len()
        # 获取数据列表
        with open(self.data_list_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        self.labels = [np.int64(line.strip().split('\t')[1]) for line in self.lines]
        if mode == 'train':
            # 获取噪声文件和混响音频
            self.noises_path = self.get_audio_path(path=aug_conf.get('noise_dir', None))
            self.reverb_path = self.get_audio_path(path=aug_conf.get('reverb_dir', None))
            logger.info(f"噪声增强的噪声音频文件数量: {len(self.noises_path)}")
            logger.info(f"混响增强音频文件数量: {len(self.reverb_path)}")
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
            # 裁剪静音
            if self.do_vad:
                audio_segment.vad()
            # 数据太短不利于训练
            if self.mode == 'train':
                if audio_segment.duration < self.min_duration:
                    logger.error(f"[{data_path}]音频太短，已跳过，最低阈值是{self.min_duration}s，"
                                 f"实际时长是{audio_segment.duration}s")
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
            if audio_segment.duration > self.max_duration:
                audio_segment.crop(duration=self.max_duration, mode=self.mode)
            samples = torch.tensor(audio_segment.samples, dtype=torch.float32)
            try:
                feature = self.audio_featurizer(samples)
            except Exception as e:
                logger.error(f"[{data_path}]特征提取失败，错误信息：{e}")
                return self.__getitem__(idx + 1 if idx < len(self.lines) - 1 else 0)
            feature = feature.squeeze(0)
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

    # 获取文件夹下的全部音频文件路径
    @staticmethod
    def get_audio_path(path):
        if path is None or not os.path.exists(path):
            return []
        paths = []
        for file in os.listdir(path):
            paths.append(os.path.join(path, file))
        return paths

    # 音频增强
    def augment_audio(self,
                      audio_segment,
                      spk_id,
                      speed_perturb=False,
                      speed_perturb_3_class=False,
                      volume_aug_prob=0.0,
                      min_gain_dBFS=-15,
                      max_gain_dBFS=15,
                      noise_dir=None,
                      noise_aug_prob=0.5,
                      min_snr_dB=10,
                      max_snr_dB=50,
                      reverb_dir=None,
                      reverb_aug_prob=0.5):
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
        if random.random() < volume_aug_prob:
            gain = random.uniform(min_gain_dBFS, max_gain_dBFS)
            audio_segment.gain_db(gain)
        # 噪声增强
        if len(self.noises_path) > 0 and random.random() < noise_aug_prob:
            # 随机选择一个noises_path中的一个
            noise_file = random.sample(self.noises_path, 1)[0]
            # 随机生成snr_dB的值
            snr_dB = random.uniform(min_snr_dB, max_snr_dB)
            # 将噪声添加到audio_segment中，并将snr_dB调整到最小值和最大值之间
            audio_segment.add_noise(noise_file, snr_dB)
        # 噪声增强
        if len(self.reverb_path) > 0 and random.random() < reverb_aug_prob:
            # 随机选择混响音频
            reverb_file = random.sample(self.noises_path, 1)[0]
            # 生成混响音效
            audio_segment.convolve(reverb_file, allow_resample=True)
        return audio_segment, spk_id
