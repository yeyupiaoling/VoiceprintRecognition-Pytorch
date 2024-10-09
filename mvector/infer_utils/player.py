import pyaudio
import soundcard
import threading
from yeaudio.audio import AudioSegment


class AudioPlayer:
    def __init__(self, audio_path):
        self.p = pyaudio.PyAudio()
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

    def callback(self, in_data, frame_count, time_info, status):
        data = self.audio_data[self.pos:self.pos + frame_count * 2]
        self.pos += frame_count * 2
        return data, pyaudio.paContinue

    def _play(self):
        self.to_pause = False
        self.playing = True
        with self.default_speaker.player(samplerate=self.sample_rate) as p:
            for i in range(int(self.pos * self.sample_rate), len(self.samples), self.block_size):
                if self.to_pause: break
                self.pos = i / self.sample_rate
                p.play(self.samples[i:i + self.block_size])
        self.playing = False

    def play(self):
        if not self.playing:
            thread = threading.Thread(target=self._play)
            thread.start()

    def pause(self):
        self.to_pause = True

    def seek(self, seconds=0.0):
        self.pos = seconds

    def current_time(self):
        return self.pos

    def close(self):
        self.p.terminate()
