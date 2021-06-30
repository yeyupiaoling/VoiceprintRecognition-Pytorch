import argparse
import functools

import numpy as np
import torch

from utils.reader import load_audio
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('audio_path1',      str,    'audio/a_1.wav',          '预测第一个音频')
add_arg('audio_path2',      str,    'audio/a_2.wav',          '预测第二个音频')
add_arg('threshold',        float,   0.7,                     '判断是否为同一个人的阈值')
add_arg('input_shape',      str,    '(1, 257, 257)',          '数据输入的形状')
add_arg('model_path',       str,    'models/infer/model',     '预测模型的路径')
args = parser.parse_args()

print_arguments(args)

device = torch.device("cuda")

# 加载模型
model = torch.jit.load(args.model_path)
model.to(device)
model.eval()


# 预测音频
def infer(audio_path):
    input_shape = eval(args.input_shape)
    data = load_audio(audio_path, mode='infer', spec_len=input_shape[2])
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    feature = model(data)
    return feature.numpy()


if __name__ == '__main__':
    # 要预测的两个人的音频文件
    feature1 = infer(args.audio_path1)[0]
    feature2 = infer(args.audio_path2)[0]
    # 对角余弦值
    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    if dist > args.threshold:
        print("%s 和 %s 为同一个人，相似度为：%f" % (args.audio_path1, args.audio_path2, dist))
    else:
        print("%s 和 %s 不是同一个人，相似度为：%f" % (args.audio_path1, args.audio_path2, dist))
