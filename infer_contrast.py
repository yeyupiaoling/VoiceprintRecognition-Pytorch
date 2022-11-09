import argparse
import functools

import yaml

from mvector.predict import MVectorPredictor
from mvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/ecapa_tdnn.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('audio_path1',      str,    'dataset/a_1.wav',          '预测第一个音频')
add_arg('audio_path2',      str,    'dataset/b_2.wav',          '预测第二个音频')
add_arg('threshold',        float,  0.6,                        '判断是否为同一个人的阈值')
add_arg('model_path',       str,    'models/{}_{}/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()

# 读取配置文件
with open(args.configs, 'r', encoding='utf-8') as f:
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)
print_arguments(args, configs)

# 获取识别器
predictor = MVectorPredictor(configs=configs,
                             model_path=args.model_path.format(configs['use_model'],
                                                               configs['preprocess_conf']['feature_method']),
                             use_gpu=args.use_gpu)

dist = predictor.contrast(args.audio_path1, args.audio_path2)
if dist > args.threshold:
    print(f"{args.audio_path1} 和 {args.audio_path2} 为同一个人，相似度为：{dist}")
else:
    print(f"{args.audio_path1} 和 {args.audio_path2} 不是同一个人，相似度为：{dist}")
