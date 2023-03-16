import argparse
import functools
import time

from mvector.trainer import MVectorTrainer
from mvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/ecapa_tdnn.yml',    "配置文件")
add_arg("use_gpu",          bool,  True,                        "是否使用GPU评估模型")
add_arg('save_image_path',  str,   'output/images/',            "保存结果图的路径")
add_arg('resume_model',     str,   'models/ecapa_tdnn_MelSpectrogram/best_model/',  "模型的路径")
args = parser.parse_args()
print_arguments(args=args)

# 获取训练器
trainer = MVectorTrainer(configs=args.configs, use_gpu=args.use_gpu)

# 开始评估
start = time.time()
tpr, fpr, eer, threshold = trainer.evaluate(resume_model=args.resume_model, save_image_path=args.save_image_path)
end = time.time()
print('评估消耗时间：{}s，threshold：{:.2f}，tpr：{:.5f}, fpr: {:.5f}, eer: {:.5f}'
      .format(int(end - start), threshold, tpr, fpr, eer))
