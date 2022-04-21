import argparse
import functools
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from modules.ecapa_tdnn import EcapaTdnn, SpeakerIdetification
from data_utils.reader import CustomDataset, collate_fn
from utils.utility import add_arguments, print_arguments, cal_accuracy_threshold

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'ecapa_tdnn',             '所使用的模型')
add_arg('num_speakers',     int,    3242,                     '分类的类别数量')
add_arg('feature_method',   str,    'melspectrogram',         '音频特征提取方法')
add_arg('list_path',        str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
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
model_path = os.path.join(args.resume, args.use_model, 'model.pdparams')
model.load_state_dict(torch.load(model_path))
print(f"成功加载模型参数和优化方法参数：{model_path}")
model.eval()


def get_all_audio_feature(list_path):
    print('开始提取全部的音频特征...')
    # 测试数据
    eval_dataset = CustomDataset(list_path,
                                 feature_method=args.feature_method,
                                 mode='eval',
                                 sr=16000,
                                 chunk_duration=3)
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=32,
                             collate_fn=collate_fn)
    accuracies = []
    features, labels = None, None
    for batch_id, (audio, label, audio_lens) in tqdm(enumerate(eval_loader)):
        output = model(audio, audio_lens)
        # 计算准确率
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        acc = np.mean((output == label).astype(int))
        accuracies.append(acc[0])
        # 获取特征
        feature = model.backbone(audio, audio_lens).data.cpu().numpy()
        features = np.concatenate((features, feature)) if features is not None else feature
        labels = np.concatenate((labels, label)) if labels is not None else label
    labels = labels.astype(np.int32)
    print('分类准确率为：{}'.format(sum(accuracies) / len(accuracies)))
    return features, labels


def main():
    features, labels = get_all_audio_feature(args.list_path)
    scores = []
    y_true = []
    print('开始两两对比音频特征...')
    for i in tqdm(range(len(features))):
        feature_1 = features[i]
        feature_1 = np.expand_dims(feature_1, 0).repeat(len(features) - i, axis=0)
        feature_2 = features[i:]
        feature_1 = torch.tensor(feature_1, dtype=torch.float32)
        feature_2 = torch.tensor(feature_2, dtype=torch.float32)
        score = F.cosine_similarity(feature_1, feature_2, dim=-1).data.cpu().numpy().tolist()
        scores.extend(score)
        y_true.extend(np.array(labels[i] == labels[i:]).astype(np.int32))
    print('找出最优的阈值和对应的准确率...')
    accuracy, threshold = cal_accuracy_threshold(scores, y_true)
    print(f'当阈值为{threshold:.2f}, 两两对比准确率最大，准确率为：{accuracy:.5f}')


if __name__ == '__main__':
    main()
