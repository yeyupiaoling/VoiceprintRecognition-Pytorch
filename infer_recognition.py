import argparse
import functools
import os
import shutil

import numpy as np
import torch

from utils.reader import load_audio
from utils.record import RecordAudio
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('input_shape',      str,    '(1, 257, 257)',          '数据输入的形状')
add_arg('threshold',        float,   0.71,                    '判断是否为同一个人的阈值')
add_arg('audio_db',         str,    'audio_db',               '音频库的路径')
add_arg('model_path',       str,    'models/resnet34.pth',    '预测模型的路径')
args = parser.parse_args()

print_arguments(args)

device = torch.device("cuda")

model = torch.jit.load(args.model_path)
model.to(device)
model.eval()

person_feature = []
person_name = []


def infer(audio_path):
    input_shape = eval(args.input_shape)
    data = load_audio(audio_path, mode='infer', spec_len=input_shape[2])
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    feature = model(data)
    return feature.data.cpu().numpy()


# 加载要识别的音频库
def load_audio_db(audio_db_path):
    audios = os.listdir(audio_db_path)
    for audio in audios:
        path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        feature = infer(path)[0]
        person_name.append(name)
        person_feature.append(feature)
        print("Loaded %s audio." % name)


def recognition(path):
    name = ''
    pro = 0
    feature = infer(path)[0]
    for i, person_f in enumerate(person_feature):
        dist = np.dot(feature, person_f) / (np.linalg.norm(feature) * np.linalg.norm(person_f))
        if dist > pro:
            pro = dist
            name = person_name[i]
    return name, pro


# 声纹注册
def register(path, user_name):
    save_path = os.path.join(args.audio_db, user_name + os.path.basename(path)[-4:])
    shutil.move(path, save_path)
    feature = infer(save_path)[0]
    person_name.append(user_name)
    person_feature.append(feature)


if __name__ == '__main__':
    load_audio_db(args.audio_db)
    record_audio = RecordAudio()

    while True:
        select_fun = int(input("请选择功能，0为注册音频到声纹库，1为执行声纹识别："))
        if select_fun == 0:
            audio_path = record_audio.record()
            name = input("请输入该音频用户的名称：")
            if name == '': continue
            register(audio_path, name)
        elif select_fun == 1:
            audio_path = record_audio.record()
            name, p = recognition(audio_path)
            if p > args.threshold:
                print("识别说话的为：%s，相似度为：%f" % (name, p))
            else:
                print("音频库没有该用户的语音")
        else:
            print('请正确选择功能')
