import argparse
import functools
import os

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
add_arg('show_plot',        bool,   True,                       '是否显示结果图像')
add_arg('search_audio_db',  bool,   True,                       '是否在音频库中搜索对应的说话人')
add_arg('threshold',        float,  0.6,                        '判断是否为同一个人的阈值')
add_arg('model_path',       str,    'models/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

if args.search_audio_db:
    assert args.audio_db_path is not None, "请指定音频库的路径"

# 获取识别器
predictor = MVectorPredictor(configs=args.configs,
                             model_path=args.model_path,
                             threshold=args.threshold,
                             audio_db_path=args.audio_db_path,
                             use_gpu=args.use_gpu)

# 进行说话人日志识别
results = predictor.speaker_diarization(args.audio_path,
                                        speaker_num=args.speaker_num,
                                        search_audio_db=args.search_audio_db)
print(f"识别结果：")
for result in results:
    print(result)

# 绘制结果图像
if args.show_plot:
    plot_speaker = PlotSpeaker(results, audio_path=args.audio_path)
    os.makedirs('output', exist_ok=True)
    plot_speaker.draw('output/speaker_diarization.png')
    plot_speaker.plot.show()
