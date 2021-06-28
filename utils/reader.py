import librosa
import numpy as np
from torch.utils import data
from aukit import remove_silence, remove_noise


# 加载并预处理音频
def load_audio(audio_path, mode='train', win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=257):
    # 读取音频数据
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    # 推理的数据要移除静音部分
    if mode == 'infer':
        wav = remove_silence(wav, sr)
        wav = remove_noise(wav, sr)
    # 数据拼接
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
    else:
        extended_wav = np.append(wav, wav[::-1])
    # 计算短时傅里叶变换
    linear = librosa.stft(extended_wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    linear_T = linear.T
    mag, _ = librosa.magphase(linear_T)
    mag_T = mag.T
    freq, freq_time = mag_T.shape
    assert freq_time >= spec_len, "非静音部分长度不能低于1.3s"
    if mode == 'train':
        # 随机裁剪
        rand_time = np.random.randint(0, freq_time - spec_len)
        spec_mag = mag_T[:, rand_time:rand_time + spec_len]
    elif mode == 'infer':
        spec_mag = mag_T
    else:
        spec_mag = mag_T[:, :spec_len]
    mean = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    spec_mag = (spec_mag - mean) / (std + 1e-5)
    spec_mag = spec_mag[np.newaxis, :]
    return spec_mag


# 数据加载器
class CustomDataset(data.Dataset):
    def __init__(self, data_list_path, model='train', spec_len=257):
        super(CustomDataset, self).__init__()
        with open(data_list_path, 'r') as f:
            self.lines = f.readlines()
        self.model = model
        self.spec_len = spec_len

    def __getitem__(self, idx):
        audio_path, label = self.lines[idx].replace('\n', '').split('\t')
        spec_mag = load_audio(audio_path, mode=self.model, spec_len=self.spec_len)
        return spec_mag, np.array(int(label), dtype=np.int64)

    def __len__(self):
        return len(self.lines)
