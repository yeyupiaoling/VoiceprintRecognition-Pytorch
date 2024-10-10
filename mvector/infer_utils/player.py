import threading

import soundcard
from yeaudio.audio import AudioSegment


class AudioPlayer:
    def __init__(self, audio_path):
        """音频播放器

        Args:
            audio_path (str): 音频文件路径
        """
        self.playing = False
        self.to_pause = False
        self.pos = 0
        self.audio_segment = AudioSegment.from_file(audio_path)
        self.audio_data = self.audio_segment.to_bytes(dtype="int16")
        self.audio_segment = AudioSegment.from_file(audio_path)
        self.audio_data = self.audio_segment.to_bytes(dtype="int16")
        self.samples = self.audio_segment.samples
        self.sample_rate = self.audio_segment.sample_rate
        self.default_speaker = soundcard.default_speaker()
        self.block_size = self.sample_rate // 2

    def _play(self):
        self.to_pause = False
        self.playing = True
        with self.default_speaker.player(samplerate=self.sample_rate) as p:
            for i in range(int(self.pos * self.sample_rate), len(self.samples), self.block_size):
                if self.to_pause: break
                self.pos = i / self.sample_rate
                p.play(self.samples[i:i + self.block_size])
        self.playing = False

    # 播放音频
    def play(self):
        if not self.playing:
            thread = threading.Thread(target=self._play)
            thread.start()

    # 暂停播放
    def pause(self):
        self.to_pause = True

    # 跳转到指定时间
    def seek(self, seconds=0.0):
        self.pos = seconds

    # 获取当前播放时间
    def current_time(self):
        return self.pos
