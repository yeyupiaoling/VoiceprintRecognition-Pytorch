# 前言
此版本为新版本，相比上一个版本，最大的变化是此版本支持pip安装，以及把预处理使用模型算子实现，这样做的好处就是可以直接使用GPU计算，大幅度提高了预处理的速度，估计预处理速度可在10~20倍。

如想使用使用旧版本，请转到[release/1.0](https://github.com/yeyupiaoling/VoiceprintRecognition_Pytorch/tree/release/1.0)，本项目使用了EcapaTdnn模型实现的声纹识别，不排除以后会支持更多模型，同时本项目也支持了多种数据预处理方法，损失函数参考了人脸识别项目的做法[PaddlePaddle-MobileFaceNets](https://github.com/yeyupiaoling/PaddlePaddle-MobileFaceNets) ,使用了ArcFace Loss，ArcFace loss：Additive Angular Margin Loss（加性角度间隔损失函数），对特征向量和权重归一化，对θ加上角度间隔m，角度间隔比余弦间隔在对角度的影响更加直接。


**欢迎大家扫码入QQ群讨论**，或者直接搜索QQ群号`1169600237`，问题答案为博主Github的ID`yeyupiaoling`。

<div align="center">
  <img src="docs/images/qq.png"/>
</div>


使用环境：

 - Anaconda 3
 - Python 3.8
 - Pytorch 1.12.1
 - Windows 10 or Ubuntu 18.04


# 模型下载


|    模型     |     预处理方法      |                          数据集                           | 类别数量  |  分类准确率  | 两两对比准确率 |   精准率   |   召回率   |                                                    模型下载地址                                                     |
|:---------:|:--------------:|:------------------------------------------------------:|:-----:|:-------:|:-------:|:-------:|:-------:|:-------------------------------------------------------------------------------------------------------------:|
| EcapaTdnn | MelSpectrogram | [中文语音语料数据集](https://github.com/fighting41love/zhvoice) | 3242  | 0.95971 | 0.99985 | 0.96294 | 0.95817 |                        [点击下载](https://download.csdn.net/download/qq_33200967/87153070)                        |
| EcapaTdnn |  Spectrogram   | [中文语音语料数据集](https://github.com/fighting41love/zhvoice) | 3242  | 0.95795 | 0.99984 | 0.96290 | 0.95786 |                        [点击下载](https://download.csdn.net/download/qq_33200967/87015334)                        |
| EcapaTdnn |      MFCC      | [中文语音语料数据集](https://github.com/fighting41love/zhvoice) | 3242  | 0.97142 | 0.99988 | 0.97131 | 0.96721 |                        [点击下载](https://download.csdn.net/download/qq_33200967/87523304)                        |
| EcapaTdnn | MelSpectrogram |                         更大的数据集                         | 6355  | 0.90020 |  0.99992 | 0.83658 | 0.85483 |  [点击下载](https://download.csdn.net/download/qq_33200967/86987829)   |   
| EcapaTdnn | MelSpectrogram |                         超大的数据集                         | 13718 |         |         |         |         |                              即将提供下载，着急可以使用旧分支[release/1.0](https://github.com/yeyupiaoling/VoiceprintRecognition_Pytorch/tree/release/1.0)       |

## 安装环境

 - 首先安装的是Pytorch的GPU版本，如果已经安装过了，请跳过。
```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

 - 安装ppvector库。
 
使用pip安装，命令如下：
```shell
python -m pip install mvector -U -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**建议源码安装**，源码安装能保证使用最新代码。
```shell
git clone https://github.com/yeyupiaoling/VoiceprintRecognition_Pytorch.git
cd VoiceprintRecognition_Pytorch/
python setup.py install
```

# 创建数据
本教程笔者使用的是[中文语音语料数据集](https://github.com/fighting41love/zhvoice) ，这个数据集一共有3242个人的语音数据，有1130000+条语音数据，下载之前要**全部解压**数据集。如果读者有其他更好的数据集，可以混合在一起使用，但最好是要用python的工具模块aukit处理音频，降噪和去除静音。

首先是创建一个数据列表，数据列表的格式为`<语音文件路径\t语音分类标签>`，创建这个列表主要是方便之后的读取，也是方便读取使用其他的语音数据集，语音分类标签是指说话人的唯一ID，不同的语音数据集，可以通过编写对应的生成数据列表的函数，把这些数据集都写在同一个数据列表中。

在`create_data.py`写下以下代码，因为[中文语音语料数据集](https://github.com/fighting41love/zhvoice) 这个数据集是mp3格式的，作者发现这种格式读取速度很慢，所以笔者把全部的mp3格式的音频转换为wav格式，这个过程可能很久。当然也可以不转换，项目也是支持的MP3格式的，只要设置参数`to_wav=False`。执行下面程序完成数据准备。
```shell
python create_data.py
```

执行上面的程序之后，会生成以下的数据格式，如果要自定义数据，参考如下数据列表，前面是音频的相对路径，后面的是该音频对应的说话人的标签，就跟分类一样。
```
dataset/zhvoice/zhmagicdata/5_895/5_895_20170614203758.wav	3238
dataset/zhvoice/zhmagicdata/5_895/5_895_20170614214007.wav	3238
dataset/zhvoice/zhmagicdata/5_941/5_941_20170613151344.wav	3239
dataset/zhvoice/zhmagicdata/5_941/5_941_20170614221329.wav	3239
dataset/zhvoice/zhmagicdata/5_941/5_941_20170616153308.wav	3239
dataset/zhvoice/zhmagicdata/5_968/5_968_20170614162657.wav	3240
dataset/zhvoice/zhmagicdata/5_968/5_968_20170622194003.wav	3240
dataset/zhvoice/zhmagicdata/5_968/5_968_20170707200554.wav	3240
dataset/zhvoice/zhmagicdata/5_970/5_970_20170616000122.wav	3241
```

# 训练模型
使用`train.py`训练模型，本项目支持多个音频预处理方式，通过`configs/ecapa_tdnn.yml`配置文件的参数`preprocess_conf.feature_method`可以指定，`MelSpectrogram`为梅尔频谱，`Spectrogram`为语谱图，`MFCC`梅尔频谱倒谱系数。通过参数`augment_conf_path`可以指定数据增强方式。训练过程中，会使用VisualDL保存训练日志，通过启动VisualDL可以随时查看训练结果，启动命令`visualdl --logdir=log --host 0.0.0.0`
```shell
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python train.py
# 多卡训练
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
```

训练输出日志：
```
[2023-02-25 11:53:53.194706 INFO   ] utils:print_arguments:13 - ----------- 额外配置参数 -----------
[2023-02-25 11:53:53.194706 INFO   ] utils:print_arguments:15 - augment_conf_path: configs/augmentation.json
[2023-02-25 11:53:53.194706 INFO   ] utils:print_arguments:15 - configs: configs/ecapa_tdnn.yml
[2023-02-25 11:53:53.194706 INFO   ] utils:print_arguments:15 - pretrained_model: None
[2023-02-25 11:53:53.194706 INFO   ] utils:print_arguments:15 - resume_model: None
[2023-02-25 11:53:53.194706 INFO   ] utils:print_arguments:15 - save_model_path: models/
[2023-02-25 11:53:53.194706 INFO   ] utils:print_arguments:15 - use_gpu: True
[2023-02-25 11:53:53.194706 INFO   ] utils:print_arguments:16 - ------------------------------------------------
[2023-02-25 11:53:53.208669 INFO   ] utils:print_arguments:18 - ----------- 配置文件参数 -----------
[2023-02-25 11:53:53.208669 INFO   ] utils:print_arguments:21 - dataset_conf:
[2023-02-25 11:53:53.208669 INFO   ] utils:print_arguments:28 - 	batch_size: 64
[2023-02-25 11:53:53.208669 INFO   ] utils:print_arguments:28 - 	chunk_duration: 3
[2023-02-25 11:53:53.208669 INFO   ] utils:print_arguments:28 - 	do_vad: False
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	min_duration: 0.5
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	num_speakers: 3242
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	num_workers: 4
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	sample_rate: 16000
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	target_dB: -20
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	test_list: dataset/test_list.txt
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	train_list: dataset/train_list.txt
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	use_dB_normalization: True
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:21 - feature_conf:
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	hop_length: 160
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	n_fft: 400
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	n_mels: 80
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	sr: 16000
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	win_length: 400
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	window: hann
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:21 - model_conf:
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	channels: [512, 512, 512, 512, 1536]
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	dilations: [1, 2, 3, 4, 1]
[2023-02-25 11:53:53.209670 INFO   ] utils:print_arguments:28 - 	kernel_sizes: [5, 3, 3, 3, 1]
[2023-02-25 11:53:53.210667 INFO   ] utils:print_arguments:28 - 	lin_neurons: 192
[2023-02-25 11:53:53.210667 INFO   ] utils:print_arguments:21 - optimizer_conf:
[2023-02-25 11:53:53.210667 INFO   ] utils:print_arguments:28 - 	learning_rate: 0.001
[2023-02-25 11:53:53.210667 INFO   ] utils:print_arguments:28 - 	weight_decay: 1e-6
[2023-02-25 11:53:53.220680 INFO   ] utils:print_arguments:21 - preprocess_conf:
[2023-02-25 11:53:53.220680 INFO   ] utils:print_arguments:28 - 	feature_method: MelSpectrogram
[2023-02-25 11:53:53.220680 INFO   ] utils:print_arguments:21 - train_conf:
[2023-02-25 11:53:53.220680 INFO   ] utils:print_arguments:28 - 	log_interval: 100
[2023-02-25 11:53:53.220680 INFO   ] utils:print_arguments:28 - 	max_epoch: 30
[2023-02-25 11:53:53.220680 INFO   ] utils:print_arguments:30 - use_model: ecapa_tdnn
[2023-02-25 11:53:53.220680 INFO   ] utils:print_arguments:31 - ------------------------------------------------
[2022-11-05 19:58:31.589525 INFO   ] augmentation:_parse_pipeline_from:126 - 数据增强配置：{'type': 'noise', 'aug_type': 'audio', 'params': {'min_snr_dB': 10, 'max_snr_dB': 50, 'repetition': 2, 'noise_dir': 'dataset/noise/'}, 'prob': 0.0}
[2022-11-05 19:58:31.589525 INFO   ] augmentation:_parse_pipeline_from:126 - 数据增强配置：{'type': 'resample', 'aug_type': 'audio', 'params': {'new_sample_rate': [8000, 32000, 44100, 48000]}, 'prob': 0.0}
[2022-11-05 19:58:31.589525 INFO   ] augmentation:_parse_pipeline_from:126 - 数据增强配置：{'type': 'speed', 'aug_type': 'audio', 'params': {'min_speed_rate': 0.9, 'max_speed_rate': 1.1, 'num_rates': 3}, 'prob': 0.0}
[2022-11-05 19:58:31.589525 INFO   ] augmentation:_parse_pipeline_from:126 - 数据增强配置：{'type': 'shift', 'aug_type': 'audio', 'params': {'min_shift_ms': -5, 'max_shift_ms': 5}, 'prob': 0.0}
[2022-11-05 19:58:31.590535 INFO   ] augmentation:_parse_pipeline_from:126 - 数据增强配置：{'type': 'volume', 'aug_type': 'audio', 'params': {'min_gain_dBFS': -15, 'max_gain_dBFS': 15}, 'prob': 0.0}
[2022-11-05 19:58:31.590535 INFO   ] augmentation:_parse_pipeline_from:126 - 数据增强配置：{'type': 'specaug', 'aug_type': 'feature', 'params': {'inplace': True, 'max_time_warp': 5, 'max_t_ratio': 0.01, 'n_freq_masks': 2, 'max_f_ratio': 0.05, 'n_time_masks': 2, 'replace_with_zero': False}, 'prob': 0.0}
[2022-11-05 19:58:31.590535 INFO   ] augmentation:_parse_pipeline_from:126 - 数据增强配置：{'type': 'specsub', 'aug_type': 'feature', 'params': {'max_t': 10, 'num_t_sub': 2}, 'prob': 0.0}
I0424 08:57:03.707505  3377 nccl_context.cc:74] init nccl context nranks: 2 local rank: 0 gpu id: 0 ring id: 0
W0424 08:57:03.930370  3377 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 10.2
W0424 08:57:03.932493  3377 device_context.cc:465] device: 0, cuDNN Version: 7.6.
I0424 08:57:05.431638  3377 nccl_context.cc:107] init nccl context nranks: 2 local rank: 0 gpu id: 0 ring id: 10
······
[2022-04-24 09:25:10.481272] Train epoch [0/30], batch: [7500/8290], loss: 9.03724, accuracy: 0.33252, lr: 0.00100000, eta: 14:58:26
[2022-04-24 09:25:32.909873] Train epoch [0/30], batch: [7600/8290], loss: 9.00004, accuracy: 0.33600, lr: 0.00100000, eta: 15:09:07
[2022-04-24 09:25:55.321806] Train epoch [0/30], batch: [7700/8290], loss: 8.96284, accuracy: 0.33950, lr: 0.00100000, eta: 15:13:13
[2022-04-24 09:26:17.836304] Train epoch [0/30], batch: [7800/8290], loss: 8.92626, accuracy: 0.34294, lr: 0.00100000, eta: 14:57:15
[2022-04-24 09:26:40.306800] Train epoch [0/30], batch: [7900/8290], loss: 8.88968, accuracy: 0.34638, lr: 0.00100000, eta: 14:51:06
[2022-04-24 09:27:02.778450] Train epoch [0/30], batch: [8000/8290], loss: 8.85430, accuracy: 0.34964, lr: 0.00100000, eta: 15:00:36
[2022-04-24 09:27:25.240278] Train epoch [0/30], batch: [8100/8290], loss: 8.81858, accuracy: 0.35294, lr: 0.00100000, eta: 14:51:58
[2022-04-24 09:27:47.690570] Train epoch [0/30], batch: [8200/8290], loss: 8.78368, accuracy: 0.35630, lr: 0.00100000, eta: 14:55:41
======================================================================
[2022-04-24 09:28:12.084404] Test 0, accuracy: 0.76057 time: 0:00:04
======================================================================
[2022-04-24 09:28:12.909394] Train epoch [1/30], batch: [0/8290], loss: 5.83753, accuracy: 0.68750, lr: 0.00099453, eta: 2 days, 3:47:48
[2022-04-24 09:28:35.346418] Train epoch [1/30], batch: [100/8290], loss: 5.80430, accuracy: 0.64527, lr: 0.00099453, eta: 15:00:01
[2022-04-24 09:28:57.873686] Train epoch [1/30], batch: [200/8290], loss: 5.78946, accuracy: 0.64218, lr: 0.00099453, eta: 14:46:39
······
```

VisualDL页面：
![VisualDL页面](./docs/images/log.jpg)


# 数据增强
本项目提供了几种音频增强操作，分布是随机裁剪，添加背景噪声，调节语速，调节音量，和SpecAugment。其中后面4种增加的参数可以在`configs/augmentation.json`修改，参数`prob`是指定该增强操作的概率，如果不想使用该增强方式，可以设置为0。要主要的是，添加背景噪声需要把多个噪声音频文件存放在`dataset/noise`，否则会跳过噪声增强
```yaml
noise:
  min_snr_dB: 10
  max_snr_dB: 30
  noise_path: "dataset/noise"
  prob: 0.5
```



# 评估模型
训练结束之后会保存预测模型，我们用预测模型来预测测试集中的音频特征，然后使用音频特征进行两两对比，阈值从0到1,步长为0.01进行控制，找到最佳的阈值并计算准确率。
```shell
python eval.py
```

输出类似如下：
```
······
------------------------------------------------
W0425 08:27:32.057426 17654 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 10.2
W0425 08:27:32.065165 17654 device_context.cc:465] device: 0, cuDNN Version: 7.6.
成功加载模型参数和优化方法参数：models/ecapa_tdnn/model.pdparams
开始提取全部的音频特征...
167it [00:15, 10.70it/s]
分类准确率为：0.9608
开始两两对比音频特征...
100%|███████████████████████████| 5332/5332 [00:05<00:00, 1027.83it/s]
找出最优的阈值和对应的准确率...
100%|███████████████████████████| 100/100 [00:06<00:00, 16.54it/s]
当阈值为0.58, 两两对比准确率最大，准确率为：0.99980
```

# 声纹对比
下面开始实现声纹对比，创建`infer_contrast.py`程序，编写`infer()`函数，在编写模型的时候，模型是有两个输出的，第一个是模型的分类输出，第二个是音频特征输出。所以在这里要输出的是音频的特征值，有了音频的特征值就可以做声纹识别了。我们输入两个语音，通过预测函数获取他们的特征数据，使用这个特征数据可以求他们的对角余弦值，得到的结果可以作为他们相识度。对于这个相识度的阈值`threshold`，读者可以根据自己项目的准确度要求进行修改。
```shell
python infer_contrast.py --audio_path1=audio/a_1.wav --audio_path2=audio/b_2.wav
```

输出类似如下：
```
-----------  Configuration Arguments -----------
audio_path1: audio/a_1.wav
audio_path2: audio/b_2.wav
feature_method: melspectrogram
resume: models/
threshold: 0.5
use_model: ecapa_tdnn
------------------------------------------------
W0425 08:29:10.006249 21121 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 10.2
W0425 08:29:10.008555 21121 device_context.cc:465] device: 0, cuDNN Version: 7.6.
成功加载模型参数和优化方法参数：models/ecapa_tdnn/model.pdparams
audio/a_1.wav 和 audio/b_2.wav 不是同一个人，相似度为：-0.09565544128417969
```

# 声纹识别
在上面的声纹对比的基础上，我们创建`infer_recognition.py`实现声纹识别。同样是使用上面声纹对比的`infer()`预测函数，通过这两个同样获取语音的特征数据。 不同的是笔者增加了`load_audio_db()`和`register()`，以及`recognition()`，第一个函数是加载声纹库中的语音数据，这些音频就是相当于已经注册的用户，他们注册的语音数据会存放在这里，如果有用户需要通过声纹登录，就需要拿到用户的语音和语音库中的语音进行声纹对比，如果对比成功，那就相当于登录成功并且获取用户注册时的信息数据。第二个函数`register()`其实就是把录音保存在声纹库中，同时获取该音频的特征添加到待对比的数据特征中。最后`recognition()`函数中，这个函数就是将输入的语音和语音库中的语音一一对比。
有了上面的声纹识别的函数，读者可以根据自己项目的需求完成声纹识别的方式，例如笔者下面提供的是通过录音来完成声纹识别。首先必须要加载语音库中的语音，语音库文件夹为`audio_db`，然后用户回车后录音3秒钟，然后程序会自动录音，并使用录音到的音频进行声纹识别，去匹配语音库中的语音，获取用户的信息。通过这样方式，读者也可以修改成通过服务请求的方式完成声纹识别，例如提供一个API供APP调用，用户在APP上通过声纹登录时，把录音到的语音发送到后端完成声纹识别，再把结果返回给APP，前提是用户已经使用语音注册，并成功把语音数据存放在`audio_db`文件夹中。
```shell
python infer_recognition.py
```

输出类似如下：
```
-----------  Configuration Arguments -----------
audio_db: audio_db
feature_method: melspectrogram
resume: models/
threshold: 0.5
use_model: ecapa_tdnn
------------------------------------------------
W0425 08:30:13.257884 23889 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 10.2
W0425 08:30:13.260191 23889 device_context.cc:465] device: 0, cuDNN Version: 7.6.
成功加载模型参数和优化方法参数：models/ecapa_tdnn/model.pdparams
Loaded 沙瑞金 audio.
Loaded 李达康 audio.
请选择功能，0为注册音频到声纹库，1为执行声纹识别：0
按下回车键开机录音，录音3秒中：
开始录音......
录音已结束!
请输入该音频用户的名称：夜雨飘零
请选择功能，0为注册音频到声纹库，1为执行声纹识别：1
按下回车键开机录音，录音3秒中：
开始录音......
录音已结束!
识别说话的为：夜雨飘零，相似度为：0.920434
```

# 其他版本
 - Tensorflow：[VoiceprintRecognition-Tensorflow](https://github.com/yeyupiaoling/VoiceprintRecognition-Tensorflow)
 - PaddlePaddle：[VoiceprintRecognition-PaddlePaddle](https://github.com/yeyupiaoling/VoiceprintRecognition-PaddlePaddle)
 - Keras：[VoiceprintRecognition-Keras](https://github.com/yeyupiaoling/VoiceprintRecognition-Keras)


# 参考资料
1. https://github.com/PaddlePaddle/PaddleSpeech
2. https://github.com/yeyupiaoling/PaddlePaddle-MobileFaceNets
3. https://github.com/yeyupiaoling/PPASR
