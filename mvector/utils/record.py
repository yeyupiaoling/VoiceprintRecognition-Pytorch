import os
import wave

import pyaudio


class RecordAudio:
    def __init__(self):
        # 录音参数
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000

        # 打开录音
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)

    def record(self, record_seconds=3, save_path=None):
        """录音

        :param record_seconds: 录音时间，默认3秒
        :param save_path: 录音保存的路径，后缀名为wav
        :return: 录音的文件路径
        """
        print("开始录音......")
        frames = []
        for i in range(0, int(self.rate / self.chunk * record_seconds)):
            data = self.stream.read(self.chunk)
            frames.append(data)

        print("录音已结束!")
        audio_data = b''.join(frames)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            wf = wave.open(save_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
        return audio_data

    def close(self):
        self.stream.close()
        self.p.close()
