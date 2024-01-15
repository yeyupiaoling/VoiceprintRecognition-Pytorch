import os
import time

import soundcard
import soundfile


class RecordAudio:
    def __init__(self, channels=1, sample_rate=16000):
        # 录音参数
        self.channels = channels
        self.sample_rate = sample_rate

        # 获取麦克风
        self.default_mic = soundcard.default_microphone()

    def record(self, record_seconds=3, save_path=None):
        """录音

        :param record_seconds: 录音时间，默认3秒
        :param save_path: 录音保存的路径，后缀名为wav
        :return: 音频的numpy数据
        """
        print("开始录音......")
        num_frames = int(record_seconds * self.sample_rate)
        start_time = time.time()
        data = self.default_mic.record(samplerate=self.sample_rate, numframes=num_frames, channels=self.channels)
        if int(time.time() - start_time) < record_seconds:
            raise Exception('录音错误，请检查录音设备，或者卸载soundfile，使用命令重新安装：'
                            'pip install git+https://github.com/bastibe/SoundCard.git')
        audio_data = data.squeeze()
        print("录音已结束!")
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            soundfile.write(save_path, data=data, samplerate=self.sample_rate)
        return audio_data
