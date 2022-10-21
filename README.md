# 前言
此版本为新版本，如想使用使用旧版本，请转到[V1.0版本](https://github.com/yeyupiaoling/VoiceprintRecognition-PaddlePaddle/tree/V1.0) ，本项目使用了EcapaTdnn模型实现的声纹识别，不排除以后会支持更多模型，同时本项目也支持了多种数据预处理方法，损失函数参考了人脸识别项目的做法[PaddlePaddle-MobileFaceNets](https://github.com/yeyupiaoling/PaddlePaddle-MobileFaceNets) ,使用了ArcFace Loss，ArcFace loss：Additive Angular Margin Loss（加性角度间隔损失函数），对特征向量和权重归一化，对θ加上角度间隔m，角度间隔比余弦间隔在对角度的影响更加直接。


**欢迎大家扫码入QQ群讨论**，或者直接搜索QQ群号`1169600237`，问题答案为博主Github的ID`yeyupiaoling`。

<div align="center">
  <img src="docs/images/qq.png"/>
</div>


使用环境：

 - Python 3.7
 - Pytorch 1.10.2


# 模型下载
|    模型     |     预处理方法      |                          数据集                           | 类别数量  | 分类准确率  | 两两对比准确率 |                                                            模型下载地址                                                             |
|:---------:|:--------------:|:------------------------------------------------------:|:-----:|:------:|:-------:|:-----------------------------------------------------------------------------------------------------------------------------:|
| EcapaTdnn | melspectrogram | [中文语音语料数据集](https://github.com/fighting41love/zhvoice) | 3242  | 0.9682 | 0.99982 |  [CSDN](https://download.csdn.net/download/qq_33200967/85270001)/[淘宝](https://item.taobao.com/item.htm?ft=t&id=688104422763)  |
| EcapaTdnn |  spectrogram   | [中文语音语料数据集](https://github.com/fighting41love/zhvoice) | 3242  | 0.9690 | 0.99982 |  [CSDN](https://download.csdn.net/download/qq_33200967/85282594)/[淘宝](https://item.taobao.com/item.htm?ft=t&id=687487364002)  |
| EcapaTdnn | melspectrogram |                         更大的数据集                         | 6355  | 0.9166 | 0.99991 | [CSDN](https://download.csdn.net/download/qq_33200967/85298248)  /[淘宝](https://item.taobao.com/item.htm?ft=t&id=688386707777) |
| EcapaTdnn |  spectrogram   |                         更大的数据集                         | 6355  | 0.9154 | 0.99990 | [CSDN](https://download.csdn.net/download/qq_33200967/85308533) /[淘宝](https://item.taobao.com/item.htm?ft=t&id=687820113874)  |
| EcapaTdnn | melspectrogram |                         超大的数据集                         | 13718 | 0.9179 | 0.99995 | [CSDN](https://download.csdn.net/download/qq_33200967/86395020) /[淘宝](https://item.taobao.com/item.htm?ft=t&id=688106534442)  |
| EcapaTdnn |  spectrogram   |                         超大的数据集                         | 13718 | 0.9344 | 0.99995 | [CSDN](https://download.csdn.net/download/qq_33200967/86398338) /[淘宝](https://item.taobao.com/item.htm?ft=t&id=688387443622)  |


# 安装环境
1. 安装Pytorch的GPU版本，如果已经安装过Pytorch，无需再次安装。
```shell
pip install torch==1.10.2
```

2. 安装其他依赖库，命令如下，注意librosa的版本是0.9.1，旧版本的梅尔频谱计算方式不一样。
```shell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

**注意：** [libsora和pyaudio安装出错解决办法](docs/faq.md)

# 创建数据
本教程笔者使用的是[中文语音语料数据集](https://github.com/fighting41love/zhvoice) ，这个数据集一共有3242个人的语音数据，有1130000+条语音数据，下载之前要全部解压数据集。如果读者有其他更好的数据集，可以混合在一起使用，但最好是要用python的工具模块aukit处理音频，降噪和去除静音。

首先是创建一个数据列表，数据列表的格式为`<语音文件路径\t语音分类标签>`，创建这个列表主要是方便之后的读取，也是方便读取使用其他的语音数据集，语音分类标签是指说话人的唯一ID，不同的语音数据集，可以通过编写对应的生成数据列表的函数，把这些数据集都写在同一个数据列表中。

在`create_data.py`写下以下代码，因为[中文语音语料数据集](https://github.com/fighting41love/zhvoice) 这个数据集是mp3格式的，作者发现这种格式读取速度很慢，所以笔者把全部的mp3格式的音频转换为wav格式，在创建数据列表之后，可能有些数据的是错误的，所以我们要检查一下，将错误的数据删除。执行下面程序完成数据准备。
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
使用`train.py`训练模型，本项目支持多个音频预处理方式，通过参数`feature_method`可以指定，`melspectrogram`为梅尔频谱，`spectrogram`为声谱图。通过参数`augment_conf_path`可以指定数据增强方式。训练过程中，会使用VisualDL保存训练日志，通过启动VisualDL可以随时查看训练结果，启动命令`visualdl --logdir=log --host 0.0.0.0`
```shell
# 单卡训练
python train.py
# 多卡训练
python train.py --gpus=0,1
```

训练输出日志：
```
-----------  Configuration Arguments -----------
augment_conf_path: configs/augment.yml
batch_size: 64
feature_method: melspectrogram
gpus: 0
learning_rate: 0.001
num_epoch: 30
num_speakers: 3242
num_workers: 4
pretrained_model: None
resume: None
save_model_dir: models/
test_list_path: dataset/test_list.txt
train_list_path: dataset/train_list.txt
use_model: ecapa_tdnn
------------------------------------------------
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
本项目提供了几种音频增强操作，分布是随机裁剪，添加背景噪声，调节语速，调节音量，和SpecAugment。其中后面4种增加的参数可以在`configs/augment.yml`修改，参数`prob`是指定该增强操作的概率，如果不想使用该增强方式，可以设置为0。要主要的是，添加背景噪声需要把多个噪声音频文件存放在`dataset/noise`，否则会跳过噪声增强
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
```shell
-----------  Configuration Arguments -----------
feature_method: melspectrogram
list_path: dataset/test_list.txt
num_speakers: 3242
resume: models/
use_model: ecapa_tdnn
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
