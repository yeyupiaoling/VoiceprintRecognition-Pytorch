import argparse
import functools
import time

from mvector.trainer import MVectorTrainer
from mvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/ecapa_tdnn.yml',    "配置文件")
add_arg("use_gpu",          bool,  True,                        "是否使用GPU评估模型")
add_arg('resume_model',     str,   'models/ecapa_tdnn_MelSpectrogram/best_model/',  "模型的路径")
args = parser.parse_args()
print_arguments(args=args)

# 获取训练器
trainer = MVectorTrainer(configs=args.configs, use_gpu=args.use_gpu)

# 开始评估
start = time.time()
loss, accuracy, precision, recall, f1_score = trainer.evaluate(resume_model=args.resume_model,
                                                               cal_threshold=True)
end = time.time()
print('评估消耗时间：{}s，loss：{:.5f}，accuracy：{:.5f}, precision: {:.5f}, recall: {:.5f}, f1_score: {:.5f}'
      .format(int(end - start), loss, accuracy, precision, recall, f1_score))
