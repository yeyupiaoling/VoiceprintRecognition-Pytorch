import argparse
import functools

from pyannote.core import Annotation
from pyannote.core import Segment
from tqdm import tqdm

from mvector.predict import MVectorPredictor
from mvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    '../../configs/cam++.yml',  '配置文件')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('data_list_path',   str,    'dataset/data_list.txt',    '要预测的音频路径列表')
add_arg('result_path',      str,    'dataset/hypotheses.rttm',  '预测结果')
add_arg('model_path',       str,    '../../models/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MVectorPredictor(configs=args.configs,
                             model_path=args.model_path,
                             use_gpu=args.use_gpu)

# 进行说话人日志识别
with open(args.data_list_path, 'r') as f_r, open(args.result_path, 'w', encoding='utf-8') as f_w:
    for line in tqdm(f_r.readlines()):
        audio_path, name = line.strip().split('\t')
        results = predictor.speaker_diarization(audio_path)

        annotation = Annotation(uri=name)
        for i, result in enumerate(results):
            annotation[Segment(result['start'], result['end']), i] = str(result['speaker'])
        f_w.write(annotation.to_rttm())
