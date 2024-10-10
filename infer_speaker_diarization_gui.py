import argparse
import functools
import os.path
import threading
import tkinter as tk
from tkinter import filedialog

from mvector.infer_utils.viewer import PlotSpeaker
from mvector.predict import MVectorPredictor
from mvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/cam++.yml',        '配置文件')
add_arg('audio_path',       str,    'dataset/test_long.wav',    '预测音频路径')
add_arg('audio_db_path',    str,    'audio_db/',                '音频库的路径')
add_arg('speaker_num',      int,    None,                       '说话人数量，提供说话人数量可以提高准确率')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('threshold',        float,  0.6,                        '判断是否为同一个人的阈值')
add_arg('model_path',       str,    'models/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)


class SpeakerDiarizationGUI:
    def __init__(self, window):
        self.window = window
        window.title("夜雨飘零说话人日志")
        self.plot_speaker = None
        self.show_plot = True
        self.search_audio_db = True
        # 添加组件
        self.label1 = tk.Label(window, text="音频路径:")
        self.label1.grid(row=0, column=0, padx=10, pady=10)
        self.entry_audio1 = tk.Entry(window, width=60)
        self.entry_audio1.grid(row=0, column=1, columnspan=2, padx=10, pady=10)
        self.btn_audio1 = tk.Button(window, text="选择", command=self.select_audio)
        self.btn_audio1.grid(row=0, column=3, padx=10, pady=10)
        self.btn_predict = tk.Button(window, text="开始识别", command=self.predict)
        self.btn_predict.grid(row=0, column=4, padx=10, pady=10)
        self.an_frame = tk.Frame(window)
        self.check_var = tk.BooleanVar(value=False)
        self.is_show_check = tk.Checkbutton(self.an_frame, text='是否显示结果图', variable=self.check_var, command=self.is_show_state)
        self.is_show_check.grid(row=0)
        self.is_show_check.select()
        self.an_frame.grid(row=1)
        self.an_frame.grid(row=2, column=1, padx=10)
        self.an_frame1 = tk.Frame(window)
        self.check_var1 = tk.BooleanVar(value=False)
        self.is_search_check = tk.Checkbutton(self.an_frame1, text='是否检索数据库', variable=self.check_var1, command=self.is_search_state)
        self.is_search_check.grid(row=0)
        self.is_search_check.select()
        self.an_frame1.grid(row=1)
        self.an_frame1.grid(row=2, column=2, padx=10)
        # 输出结果文本框
        self.result_label = tk.Label(self.window, text="输出结果：")
        self.result_label.grid(row=3, column=0, padx=10, pady=10)
        self.result_text = tk.Text(self.window, width=60, height=20)
        self.result_text.grid(row=3, column=1, columnspan=2, padx=10, pady=10)

        # 预测器
        self.predictor = MVectorPredictor(configs=args.configs,
                                          model_path=args.model_path,
                                          threshold=args.threshold,
                                          audio_db_path=args.audio_db_path,
                                          use_gpu=args.use_gpu)

    def is_show_state(self):
        self.show_plot = self.check_var.get()

    def is_search_state(self):
        self.search_audio_db = self.check_var1.get()

    def select_audio(self):
        filename = filedialog.askopenfilename(initialdir='./dataset')
        self.entry_audio1.delete(0, tk.END)
        self.entry_audio1.insert(tk.END, filename)

    def predict(self):
        if self.plot_speaker:
            self.plot_speaker.plot.close()
        self.plot_speaker = None
        audio_path = self.entry_audio1.get()
        if audio_path is None or len(audio_path) == 0: return
        print(f'选择音频路径：{audio_path}')
        # 进行说话人日志识别
        results = self.predictor.speaker_diarization(audio_path,
                                                     speaker_num=args.speaker_num,
                                                     search_audio_db=self.search_audio_db)
        self.result_text.delete('1.0', 'end')
        for result in results:
            self.result_text.insert(tk.END, f"{result}\n")

        if self.show_plot:
            threading.Thread(target=self.show_result(results), args=(results,)).start()

    def show_result(self, results):
        self.plot_speaker = PlotSpeaker(results, audio_path=args.audio_path)
        os.makedirs('output', exist_ok=True)
        self.plot_speaker.draw('output/speaker_diarization.png')
        self.plot_speaker.plot.show()
        self.plot_speaker = None


if __name__ == '__main__':
    root = tk.Tk()
    app = SpeakerDiarizationGUI(root)
    root.mainloop()
