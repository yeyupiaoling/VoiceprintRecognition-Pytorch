import argparse
import functools

from mvector.predict import MVectorPredictor
from mvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/cam++.yml',        '配置文件')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('show_plot',        bool,   True,                       '是否显示结果图像')
add_arg('audio_path',       str,    'dataset/test_long.wav',    '预测音频路径')
add_arg('model_path',       str,    'models/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MVectorPredictor(configs=args.configs,
                             model_path=args.model_path,
                             use_gpu=args.use_gpu)

result = predictor.speaker_diarization(args.audio_path, show_plot=args.show_plot)
print(result)
