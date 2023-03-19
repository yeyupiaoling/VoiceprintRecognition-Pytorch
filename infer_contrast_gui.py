import os.path
import tkinter as tk
from tkinter import filedialog, messagebox
import functools
import argparse
from mvector.predict import MVectorPredictor
from mvector.utils.utils import add_arguments, print_arguments


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/ecapa_tdnn.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('model_path',       str,    'models/ecapa_tdnn_MelSpectrogram/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)


class VoiceContrastGUI:
    def __init__(self, master):
        self.master = master
        master.title("夜雨飘零声纹对比")
        # 添加组件
        self.label1 = tk.Label(master, text="音频1路径:")
        self.label1.grid(row=0, column=0, padx=10, pady=10)
        self.entry_audio1 = tk.Entry(master, width=50)
        self.entry_audio1.grid(row=0, column=1, columnspan=2, padx=10, pady=10)
        self.btn_audio1 = tk.Button(master, text="选择", command=self.select_audio1)
        self.btn_audio1.grid(row=0, column=3, padx=10, pady=10)
        self.label2 = tk.Label(master, text="音频2路径:")
        self.label2.grid(row=1, column=0, padx=10, pady=10)
        self.entry_audio2 = tk.Entry(master, width=50)
        self.entry_audio2.grid(row=1, column=1, columnspan=2, padx=10, pady=10)
        self.btn_audio2 = tk.Button(master, text="选择", command=self.select_audio2)
        self.btn_audio2.grid(row=1, column=3, padx=10, pady=10)
        self.label3 = tk.Label(master, text="判断阈值:")
        self.label3.grid(row=2, column=0, padx=10, pady=10)
        self.entry_threshold = tk.Entry(master, width=50)
        self.entry_threshold.insert(0, "0.6")
        self.entry_threshold.grid(row=2, column=1, columnspan=2, padx=10, pady=10)
        self.btn_predict = tk.Button(master, text="开始判断", command=self.predict)
        self.btn_predict.grid(row=3, column=1, padx=10, pady=10)
        self.btn_quit = tk.Button(master, text="退出", command=self.quit)
        self.btn_quit.grid(row=3, column=2, padx=10, pady=10)
        # 预测器
        self.predictor = MVectorPredictor(configs=args.configs, model_path=args.model_path, use_gpu=args.use_gpu)

    def select_audio1(self):
        filename = filedialog.askopenfilename()
        self.entry_audio1.delete(0, tk.END)
        self.entry_audio1.insert(tk.END, filename)

    def select_audio2(self):
        filename = filedialog.askopenfilename()
        self.entry_audio2.delete(0, tk.END)
        self.entry_audio2.insert(tk.END, filename)

    def predict(self):
        audio_path1 = self.entry_audio1.get()
        audio_path2 = self.entry_audio2.get()
        threshold = float(self.entry_threshold.get())
        if not audio_path1 or not audio_path2:
            messagebox.showerror("错误", "请选择两个音频文件")
            return
        try:
            dist = self.predictor.contrast(audio_path1, audio_path2)
        except Exception as e:
            messagebox.showerror("错误", "预测失败，请检查音频文件格式是否正确")
            return
        if dist > threshold:
            messagebox.showinfo("结果", f"{os.path.basename(audio_path1)} 和 {os.path.basename(audio_path2)} 为同一个人，"
                                      f"相似度为：{dist:.5f}")
        else:
            messagebox.showinfo("结果", f"{os.path.basename(audio_path1)} 和 {os.path.basename(audio_path2)} 不是同一个人，"
                                      f"相似度为：{dist:.5f}")

    def quit(self):
        self.master.quit()


if __name__ == '__main__':
    root = tk.Tk()
    app = VoiceContrastGUI(root)
    root.mainloop()
