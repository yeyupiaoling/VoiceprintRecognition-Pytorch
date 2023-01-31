import numpy as np
from torch.utils.data import Dataset

from mvector.data_utils.audio import AudioSegment
from mvector.data_utils.augmentor.augmentation import AugmentationPipeline
from mvector.utils.logger import setup_logger

logger = setup_logger(__name__)


# 音频数据加载器
class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 do_vad=True,
                 chunk_duration=3,
                 min_duration=0.5,
                 augmentation_config='{}',
                 mode='train',
                 sample_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20):
        super(CustomDataset, self).__init__()
        self.do_vad = do_vad
        self.chunk_duration = chunk_duration
        self.min_duration = min_duration
        self.mode = mode
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self._augmentation_pipeline = AugmentationPipeline(augmentation_config=augmentation_config)
        # 获取数据列表
        with open(data_list_path, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, idx):
        # 分割音频路径和标签
        audio_path, label = self.lines[idx].replace('\n', '').split('\t')
        # 读取音频
        audio_segment = AudioSegment.from_file(audio_path)
        # 裁剪静音
        if self.do_vad:
            audio_segment.vad()
        # 数据太短不利于训练
        if self.mode == 'train':
            if audio_segment.num_samples < int(self.min_duration * audio_segment.sample_rate):
                return self.__getitem__(idx + 1 if idx < len(self.lines) - 1 else 0)
        # 重采样
        if audio_segment.sample_rate != self._target_sample_rate:
            audio_segment.resample(self._target_sample_rate)
        # decibel normalization
        if self._use_dB_normalization:
            audio_segment.normalize(target_db=self._target_dB)
        # 音频增强
        self._augmentation_pipeline.transform_audio(audio_segment)
        # 对小于训练长度的复制补充
        num_chunk_samples = int(self.chunk_duration * audio_segment.sample_rate)
        if audio_segment.num_samples < num_chunk_samples:
            shortage = num_chunk_samples - audio_segment.num_samples
            audio_segment.pad_silence(duration=float(shortage / audio_segment.sample_rate * 1.1), sides='end')
        # 裁剪需要的数据
        audio_segment.crop(duration=self.chunk_duration, mode=self.mode)
        return np.array(audio_segment.samples, dtype=np.float32), np.array(int(label), dtype=np.int64)

    def __len__(self):
        return len(self.lines)
