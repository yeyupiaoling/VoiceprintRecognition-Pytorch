# This implementation is adapted from https://github.com/taylorlu/Speaker-Diarization
import matplotlib.pyplot as plot

from mvector.infer_utils.player import AudioPlayer


class PlotSpeaker:
    def __init__(self, speakers_data, audio_path=None, title="speaker-diarization", gui=False, size=(14, 6)):
        self.rect_color = (0.0, 0.6, 1.0, 1.0)
        self.rect_selected_color = (0.75, 0.75, 0, 1.0)  # 'y'
        self.cluster_colors = [(0.0, 0.6, 1.0, 1.0), (0.0, 1.0, 0.6, 1.0), (0.6, 0.0, 1.0, 1.0),
                               (0.6, 1.0, 0.0, 1.0), (1.0, 0.0, 0.6, 1.0), (1.0, 0.6, 0.0, 1.0)]
        self.gui = gui
        self.title = title
        self.fig = plot.figure(figsize=size, facecolor='white', tight_layout=True)
        self.plot = plot

        self.ax = self.fig.add_subplot(1, 1, 1)
        if self.gui:
            self.fig.canvas.mpl_connect('key_press_event', self._on_keypress)
            self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.height = 5
        self.maxx = 0
        self.audio = None
        if audio_path is not None and self.gui:
            self.audio = AudioPlayer(audio_path)
            self.timer = self.fig.canvas.new_timer(interval=500)
            self.timer.add_callback(self._update_timeline)
            self.timer.start()

        self.timeline = self.ax.plot([0, 0], [0, 0], color='r')[-1]
        segment_data = dict()
        for data in speakers_data:
            start, end, speaker = data['start'], data['end'], data['speaker']
            if speaker not in segment_data:
                segment_data[speaker] = []
            segment_data[speaker].append(dict(start=start, end=end))
        self.speakers_data = segment_data

    def _update_timeline(self):
        """
        Update the timeline given the position in the audio player

        """
        if self.audio is not None and self.audio.playing:
            t = self.audio.current_time()
            self._draw_timeline(t)
            self.fig.canvas.draw()

    def _draw_timeline(self, t):
        """
        Draw the timeline a position t
        :param t: in second, a float

        """
        min_y, max_y = self.ax.get_ylim()
        self.timeline.set_data([t, t], [min_y, max_y])
        self._draw_info(t)

    @staticmethod
    def _draw_info(t):
        """
        Draw information on segment and timestamp
        :param t: a float
        :return:
        """
        h = int(t) // 3600
        t %= 3600
        m = int(t) // 60
        s = int(t % 60)
        plot.xlabel(f'time: {h:02}:{m:02}:{s:02}')

    def draw(self, save_path=None):
        y = 0
        labels_pos = []
        labels = []
        for i, cluster in enumerate(sorted(self.speakers_data.keys())):
            labels.append(cluster)
            labels_pos.append(y + self.height // 2)
            for row in self.speakers_data[cluster]:
                x = row['start']
                w = row['end'] - row['start']
                self.maxx = max(self.maxx, row['end'])
                c = self.cluster_colors[i % len(self.cluster_colors)]
                rect = plot.Rectangle((x, y), w, self.height, color=c)
                self.ax.add_patch(rect)
            y += self.height
        if self.gui:
            plot.xlim([0, min(600, self.maxx)])
        else:
            plot.xlim([0, self.maxx])

        plot.ylim([0, y])
        plot.yticks(labels_pos, labels)
        for _ in self.speakers_data:
            self.ax.plot([0, self.maxx], [y, y], linestyle=':', color='#AAAAAA')
            y -= self.height

        plot.title(self.title)
        if self.gui:
            self._draw_info(0)
        plot.tight_layout()
        if save_path is not None:
            plot.savefig(save_path)

    # 键盘点击事件处理函数
    def _on_keypress(self, event):
        if event.key == ' ' and self.audio is not None:
            if self.audio.playing:
                self.audio.pause()
            else:
                self.audio.play()
        self.fig.canvas.draw()

    # 鼠标点击事件处理函数
    def _on_click(self, event):
        if event.xdata is not None:
            if self.audio is not None:
                self.audio.pause()
                self.audio.seek(event.xdata)
            self._draw_timeline(event.xdata)
            self.fig.canvas.draw()
