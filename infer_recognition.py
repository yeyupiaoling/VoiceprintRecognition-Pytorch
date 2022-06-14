import argparse
import functools
import os
import shutil

import numpy as np
import torch

from modules.ecapa_tdnn import EcapaTdnn, SpeakerIdetification
from data_utils.reader import load_audio, CustomDataset
from utils.record import RecordAudio
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'ecapa_tdnn',             '所使用的模型')
add_arg('threshold',        float,   0.5,                     '判断是否为同一个人的阈值')
add_arg('audio_db',         str,    'audio_db',               '音频库的路径')
add_arg('feature_method',   str,    'melspectrogram',         '音频特征提取方法', choices=['melspectrogram', 'spectrogram'])
add_arg('resume',           str,    'models/',                '模型文件夹路径')
args = parser.parse_args()
print_arguments(args)

dataset = CustomDataset(data_list_path=None, feature_method=args.feature_method)
# 获取模型
if args.use_model == 'ecapa_tdnn':
    ecapa_tdnn = EcapaTdnn(input_size=dataset.input_size)
    model = SpeakerIdetification(backbone=ecapa_tdnn)
else:
    raise Exception(f'{args.use_model} 模型不存在！')
# 指定使用设备
device = torch.device("cuda")
model.to(device)
# 加载模型
model_path = os.path.join(args.resume, args.use_model, 'model.pth')
model_dict = model.state_dict()
param_state_dict = torch.load(model_path)
for name, weight in model_dict.items():
    if name in param_state_dict.keys():
        if list(weight.shape) != list(param_state_dict[name].shape):
            param_state_dict.pop(name, None)
model.load_state_dict(param_state_dict, strict=False)
print(f"成功加载模型参数和优化方法参数：{model_path}")
model.eval()

person_feature = []
person_name = []


# 执行识别
def infer(audio_path):
    data = load_audio(audio_path, mode='infer', feature_method=args.feature_method)
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    feature = model.backbone(data)
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
        print(f"Loaded {name} audio.")


# 声纹识别
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
                print(f"识别说话的为：{name}，相似度为：{p}")
            else:
                print("音频库没有该用户的语音")
        else:
            print('请正确选择功能')
