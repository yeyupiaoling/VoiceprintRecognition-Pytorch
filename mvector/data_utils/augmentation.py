# @Time    : 2024-08-30
# @Author  : yeyupiaoling
import os
import random

import torch
import torch.nn as nn
from loguru import logger
from yeaudio.audio import AudioSegment


class SpeedPerturbAugmentor(object):
    def __init__(self, prob=1.0, speed_perturb_3_class=False, num_speakers=None):
        self.speeds = [1.0, 0.9, 1.1]
        self.prob = prob
        self.num_speakers = num_speakers
        self.speed_perturb_3_class = speed_perturb_3_class
        if self.speed_perturb_3_class:
            assert self.num_speakers is not None, "使用语速三类语速增强的话，需要设置num_speakers参数"

    def __call__(self, audio_segment: AudioSegment, spk_id: int):
        if random.random() < self.prob:
            speed_idx = random.randint(0, 2)
            speed_rate = self.speeds[speed_idx]
            if speed_rate != 1.0:
                audio_segment.change_speed(speed_rate)
            # 注意使用语速增强分类数量会大三倍
            if self.speed_perturb_3_class:
                spk_id = spk_id + self.num_speakers * speed_idx
        return audio_segment, spk_id


class VolumePerturbAugmentor(object):
    def __init__(self, prob=0.0, min_gain_dBFS=-15, max_gain_dBFS=15):
        self.prob = prob
        self.min_gain_dBFS = min_gain_dBFS
        self.max_gain_dBFS = max_gain_dBFS

    def __call__(self, audio_segment: AudioSegment):
        if random.random() < self.prob:
            gain = random.uniform(self.min_gain_dBFS, self.max_gain_dBFS)
            audio_segment.gain_db(gain)
        return audio_segment


class NoisePerturbAugmentor(object):
    def __init__(self, noise_dir='', prob=0.5, min_snr_dB=10, max_snr_dB=50):
        self.prob = prob
        self.min_snr_dB = min_snr_dB
        self.max_snr_dB = max_snr_dB
        self.noises_path = self.get_audio_path(path=noise_dir)
        logger.info(f"噪声增强的噪声音频文件数量: {len(self.noises_path)}")

    def __call__(self, audio_segment: AudioSegment):
        if len(self.noises_path) > 0 and random.random() < self.prob:
            # 随机选择一个noises_path中的一个
            noise_file = random.sample(self.noises_path, 1)[0]
            # 随机生成snr_dB的值
            snr_dB = random.uniform(self.min_snr_dB, self.max_snr_dB)
            # 将噪声添加到audio_segment中，snr_dB是噪声的增益
            audio_segment.add_noise(noise_file, snr_dB)
        return audio_segment

    # 获取文件夹下的全部音频文件路径
    @staticmethod
    def get_audio_path(path):
        if path is None or not os.path.exists(path):
            return []
        paths = []
        for file in os.listdir(path):
            paths.append(os.path.join(path, file))
        return paths


class ReverbPerturbAugmentor(object):
    def __init__(self, reverb_dir='', prob=0.5):
        self.prob = prob
        self.reverb_path = self.get_audio_path(path=reverb_dir)
        logger.info(f"混响增强音频文件数量: {len(self.reverb_path)}")

    def __call__(self, audio_segment: AudioSegment):
        if len(self.reverb_path) > 0 and random.random() < self.prob:
            # 随机选择混响音频
            reverb_file = random.sample(self.reverb_path, 1)[0]
            # 生成混响音效
            audio_segment.reverb(reverb_file)
        return audio_segment

    # 获取文件夹下的全部音频文件路径
    @staticmethod
    def get_audio_path(path):
        if path is None or not os.path.exists(path):
            return []
        paths = []
        for file in os.listdir(path):
            paths.append(os.path.join(path, file))
        return paths


class SpecAugmentor(nn.Module):

    def __init__(self, prob=1.0, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.prob = prob
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
        if random.random() < self.prob:
            x = self.freq_mask(x)
            x = self.time_mask(x)
        return x
