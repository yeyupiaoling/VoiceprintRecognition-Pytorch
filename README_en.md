[简体中文](./README.md) | English

# Voiceprint recognition system based on Pytorch

![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/yeyupiaoling/VoiceprintRecognition-Pytorch)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/VoiceprintRecognition-Pytorch)
![GitHub](https://img.shields.io/github/license/yeyupiaoling/VoiceprintRecognition-Pytorch)
![支持系统](https://img.shields.io/badge/支持系统-Win/Linux/MAC-9cf)

**Disclaimer, this document was obtained through machine translation, please check the original document [here](./README.md).**


This branch is version 1.0, if you want to use the previous version 0.3 please [0. X branch](https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch/tree/release/0.x). This project uses a variety of advanced voiceprint recognition models such as EcapaTdnn, ResNetSE, ERes2Net, CAM++, etc. It is not excluded that more models will be supported in the future. At the same time, this project also supports MelSpectrogram, Spectrogram, MFCC, Fbank and other data preprocessing methods, using ArcFace Loss, ArcFace loss: Additive Angular Margin Loss, corresponding to AAMLoss in the project, normalizes the feature vectors and weights, and adds an Angle margin m to θ. The Angle margin has a more direct effect on the Angle than the cosine margin. In addition, Various loss functions such as AMLoss, ARMLoss, CELoss are also supported.


Environment：

 - Anaconda 3
 - Python 3.11
 - Pytorch 2.0.1
 - Windows 10 or Ubuntu 18.04

# Project Features

1. Supporting models: EcapaTdnn、TDNN、Res2Net、ResNetSE、ERes2Net、CAM++
2. Supporting pooling: AttentiveStatsPool(ASP)、SelfAttentivePooling(SAP)、TemporalStatisticsPooling(TSP)、TemporalAveragePooling(TAP)、TemporalStatsPool(TSTP)
3. Supporting Loss: AAMLoss、SphereFace2、AMLoss、ARMLoss、CELoss
4. Support preprocessing methods: MelSpectrogram、Spectrogram、MFCC、Fbank

**Model Paper：**

- EcapaTdnn：[ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](https://arxiv.org/abs/2005.07143v3)
- PANNS：[PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211v5)
- TDNN：[Prediction of speech intelligibility with DNN-based performance measures](https://arxiv.org/abs/2203.09148)
- Res2Net：[Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)
- ResNetSE：[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- CAMPPlus：[CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking](https://arxiv.org/abs/2303.00332v3)
- ERes2Net：[An Enhanced Res2Net with Local and Global Feature Fusion for Speaker Verification](https://arxiv.org/abs/2305.12838v1)


# Download Model

|   Model    | Params(M) | Preprocessing method |              Dataset               | train speakers | threshold |   EER   | MinDCF  | 
|:----------:|:---------:|:--------------------:|:----------------------------------:|:--------------:|:---------:|:-------:|:-------:| 
|   CAM++    |    7.5    |        Fbank         | [CN-Celeb](http://openslr.org/82/) |      2796      |   0.26    | 0.09557 | 0.53516 |        
|  ERes2Net  |    8.2    |        Fbank         | [CN-Celeb](http://openslr.org/82/) |      2796      |   0.19    | 0.09980 | 0.52352 |        
|  ResNetSE  |    9.4    |        Fbank         | [CN-Celeb](http://openslr.org/82/) |      2796      |   0.20    | 0.10149 | 0.55185 |        
| EcapaTdnn  |    6.7    |        Fbank         | [CN-Celeb](http://openslr.org/82/) |      2796      |   0.24    | 0.10163 | 0.56543 |   
|    TDNN    |    3.2    |        Fbank         | [CN-Celeb](http://openslr.org/82/) |      2796      |   0.23    | 0.12182 | 0.62141 |     
|  Res2Net   |    6.6    |        Fbank         | [CN-Celeb](http://openslr.org/82/) |      2796      |   0.22    | 0.14390 | 0.67961 |     
|   CAM++    |    7.5    |        Fbank         |               更大数据集                |      2W+       |   0.33    | 0.07874 | 0.52524 |
|  ERes2Net  |    8.2    |        Fbank         |               其他数据集                |      20W       |   0.36    | 0.02936 | 0.18355 |  
| ERes2NetV2 |   56.2    |        Fbank         |               其他数据集                |      20W+      |   0.36    | 0.03847 | 0.24301 |
|   CAM++    |    7.5    |        Fbank         |               其他数据集                |      20W       |   0.29    | 0.04765 | 0.31436 |        

Explain:

1. [CN-Celeb Test](https://aistudio.baidu.com/aistudio/datasetdetail/233361), which contains 196 speakers.
2. Triple the classification size using speech rate augmentation`speed_perturb_3_class: True`.
3. The number of parameters does not include the number of parameters of the classifier.


### VoxCeleb1&2数据

|     模型     | Params(M) | Preprocessing method |   Dataset   | train speakers | threshold |   EER   | MinDCF  |
|:----------:|:---------:|:--------------------:|:-----------:|:--------------:|:---------:|:-------:|:-------:|
|   CAM++    |    6.8    |        Fbank         | VoxCeleb1&2 |      7205      |   0.23    | 0.02659 | 0.18604 |
|  ERes2Net  |    6.6    |        Fbank         | VoxCeleb1&2 |      7205      |   0.23    | 0.03648 | 0.25508 | 
|  ResNetSE  |    7.8    |        Fbank         | VoxCeleb1&2 |      7205      |   0.23    | 0.03668 | 0.27881 | 
| EcapaTdnn  |    6.1    |        Fbank         | VoxCeleb1&2 |      7205      |   0.26    | 0.02610 | 0.18008 |
|    TDNN    |    2.6    |        Fbank         | VoxCeleb1&2 |      7205      |   0.26    | 0.03963 | 0.31433 |
|  Res2Net   |    5.0    |        Fbank         | VoxCeleb1&2 |      7205      |   0.20    | 0.04290 | 0.41416 |
|   CAM++    |    6.8    |        Fbank         |    更大数据集    |      2W+       |   0.28    | 0.03182 | 0.23731 | 
|  ERes2Net  |   55.1    |        Fbank         |    其他数据集    |      20W+      |   0.53    | 0.08904 | 0.62130 | 
| ERes2NetV2 |   56.2    |        Fbank         |    其他数据集    |      20W+      |   0.52    | 0.08649 | 0.64193 |
|   CAM++    |    6.8    |        Fbank         |    其他数据集    |      20W+      |   0.49    | 0.10334 | 0.71200 | 

Explain：

1. [VoxCeleb1&2 Test](https://aistudio.baidu.com/aistudio/datasetdetail/255977), which contains 158 speakers.
2. Triple the classification size using speech rate augmentation`speed_perturb_3_class: True`.
3. The number of parameters does not include the number of parameters of the classifier.


## Installation Environment

 - The GPU version of Pytorch will be installed first, please skip it if you already have it installed.
```shell
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

 - Install ppvector.
 
Install it using pip with the following command:
```shell
python -m pip install mvector -U -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Source installation is recommended**, which ensures that the latest code is used.
```shell
git clone https://github.com/yeyupiaoling/VoiceprintRecognition_Pytorch.git
cd VoiceprintRecognition_Pytorch/
python setup.py install
```

# Create Data
The author used [CN-Celeb](https://openslr.elda.org/resources/82) for this tutorial. This dataset has a total of about 3000 people's voice data, and there are 65W+ voice data. After downloading, you need to unzip the dataset to the 'dataset' directory. Also need to download [CN-Celeb Test](https://aistudio.baidu.com/aistudio/datasetdetail/233361). If you have other better datasets, you can mix them up, but it's best to use python's aukit tool module for audio processing, noise reduction, and de-muting.

The format of the data list is `<voice_file_path\tspeech_classification_label>`. The creation of this list is mainly for the convenience of later reading, but also for the convenience of reading and using other speech data sets. Speech classification label refers to the unique ID of the speaker. Put these data sets in the same data list.

Execute `create_data.py` to prepare the data.
```shell
python create_data.py
```

After executing the above program, the following data format will be generated, and if you want to customize the data, refer to the following list of data, which is preceded by the relative path of the audio and followed by the label of the speaker for that audio, just like classification. **A note on custom datasets**, the test list ID doesn't have to be the same as the training ID, meaning the test speaker doesn't have to be in the training set, just make sure the test list has the same ID for the same person.
```
dataset/CN-Celeb2_flac/data/id11999/recitation-03-019.flac      2795
dataset/CN-Celeb2_flac/data/id11999/recitation-10-023.flac      2795
dataset/CN-Celeb2_flac/data/id11999/recitation-06-025.flac      2795
dataset/CN-Celeb2_flac/data/id11999/recitation-04-014.flac      2795
dataset/CN-Celeb2_flac/data/id11999/recitation-06-030.flac      2795
dataset/CN-Celeb2_flac/data/id11999/recitation-10-032.flac      2795
dataset/CN-Celeb2_flac/data/id11999/recitation-06-028.flac      2795
dataset/CN-Celeb2_flac/data/id11999/recitation-10-031.flac      2795
dataset/CN-Celeb2_flac/data/id11999/recitation-05-003.flac      2795
dataset/CN-Celeb2_flac/data/id11999/recitation-04-017.flac      2795
dataset/CN-Celeb2_flac/data/id11999/recitation-10-016.flac      2795
dataset/CN-Celeb2_flac/data/id11999/recitation-09-001.flac      2795
dataset/CN-Celeb2_flac/data/id11999/recitation-05-010.flac      2795
```

# Change preprocessing methods

By default, the MelSpectrogram preprocessing method is used in the configuration file. If you want to use other preprocessing methods, you can modify the following installation in the configuration file, and the specific value can be modified according to your own situation. If it's not clear how to set the parameters, you can remove that section and just use the default values.

```yaml
preprocess_conf:
  # 音频预处理方法，支持：MelSpectrogram、Spectrogram、MFCC、Fbank
  feature_method: 'MelSpectrogram'
  # 设置API参数，更参数查看对应API，不清楚的可以直接删除该部分，直接使用默认值
  method_args:
    sample_rate: 16000
    n_fft: 1024
    hop_length: 320
    win_length: 1024
    f_min: 50.0
    f_max: 14000.0
    n_mels: 64
```

# Train

Using `train.py` to train the model, this project supports multiple audio preprocessing methods, which can be specified by the `preprocess_conf.feature_method` parameter in the `configs/ecapa_tdnn.yml` configuration file. `MelSpectrogram` for MEL spectrum, `Spectrogram` for spectrogram, `MFCC` for MEL spectrum cepstral coefficient, etc. The data augmentation can be specified using the `augment_conf_path` argument. During the training process, VisualDL will be used to save the training logs. You can view the training results at any time by starting VisualDL with the command `visualdl --logdir=log --host 0.0.0.0`
```shell
# Single GPU training
CUDA_VISIBLE_DEVICES=0 python train.py
# Multi GPU training
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
```

Train log:
```
[2023-08-05 09:52:06.497988 INFO   ] utils:print_arguments:13 - ----------- 额外配置参数 -----------
[2023-08-05 09:52:06.498094 INFO   ] utils:print_arguments:15 - configs: configs/ecapa_tdnn.yml
[2023-08-05 09:52:06.498149 INFO   ] utils:print_arguments:15 - do_eval: True
[2023-08-05 09:52:06.498191 INFO   ] utils:print_arguments:15 - local_rank: 0
[2023-08-05 09:52:06.498230 INFO   ] utils:print_arguments:15 - pretrained_model: None
[2023-08-05 09:52:06.498269 INFO   ] utils:print_arguments:15 - resume_model: None
[2023-08-05 09:52:06.498306 INFO   ] utils:print_arguments:15 - save_model_path: models/
[2023-08-05 09:52:06.498342 INFO   ] utils:print_arguments:15 - use_gpu: True
[2023-08-05 09:52:06.498378 INFO   ] utils:print_arguments:16 - ------------------------------------------------
[2023-08-05 09:52:06.513761 INFO   ] utils:print_arguments:18 - ----------- 配置文件参数 -----------
[2023-08-05 09:52:06.513906 INFO   ] utils:print_arguments:21 - dataset_conf:
[2023-08-05 09:52:06.513957 INFO   ] utils:print_arguments:24 -         dataLoader:
[2023-08-05 09:52:06.513995 INFO   ] utils:print_arguments:26 -                 batch_size: 64
[2023-08-05 09:52:06.514031 INFO   ] utils:print_arguments:26 -                 num_workers: 4
[2023-08-05 09:52:06.514066 INFO   ] utils:print_arguments:28 -         do_vad: False
[2023-08-05 09:52:06.514101 INFO   ] utils:print_arguments:28 -         enroll_list: dataset/enroll_list.txt
[2023-08-05 09:52:06.514135 INFO   ] utils:print_arguments:24 -         eval_conf:
[2023-08-05 09:52:06.514169 INFO   ] utils:print_arguments:26 -                 batch_size: 1
[2023-08-05 09:52:06.514203 INFO   ] utils:print_arguments:26 -                 max_duration: 20
[2023-08-05 09:52:06.514237 INFO   ] utils:print_arguments:28 -         max_duration: 3
[2023-08-05 09:52:06.514274 INFO   ] utils:print_arguments:28 -         min_duration: 0.5
[2023-08-05 09:52:06.514308 INFO   ] utils:print_arguments:28 -         noise_aug_prob: 0.2
[2023-08-05 09:52:06.514342 INFO   ] utils:print_arguments:28 -         noise_dir: dataset/noise
[2023-08-05 09:52:06.514374 INFO   ] utils:print_arguments:28 -         num_speakers: 3242
[2023-08-05 09:52:06.514408 INFO   ] utils:print_arguments:28 -         sample_rate: 16000
[2023-08-05 09:52:06.514441 INFO   ] utils:print_arguments:28 -         speed_perturb: True
[2023-08-05 09:52:06.514475 INFO   ] utils:print_arguments:28 -         target_dB: -20
[2023-08-05 09:52:06.514508 INFO   ] utils:print_arguments:28 -         train_list: dataset/train_list.txt
[2023-08-05 09:52:06.514542 INFO   ] utils:print_arguments:28 -         trials_list: dataset/trials_list.txt
[2023-08-05 09:52:06.514575 INFO   ] utils:print_arguments:28 -         use_dB_normalization: True
[2023-08-05 09:52:06.514609 INFO   ] utils:print_arguments:21 - loss_conf:
[2023-08-05 09:52:06.514643 INFO   ] utils:print_arguments:24 -         args:
[2023-08-05 09:52:06.514678 INFO   ] utils:print_arguments:26 -                 easy_margin: False
[2023-08-05 09:52:06.514713 INFO   ] utils:print_arguments:26 -                 margin: 0.2
[2023-08-05 09:52:06.514746 INFO   ] utils:print_arguments:26 -                 scale: 32
[2023-08-05 09:52:06.514779 INFO   ] utils:print_arguments:24 -         margin_scheduler_args:
[2023-08-05 09:52:06.514814 INFO   ] utils:print_arguments:26 -                 final_margin: 0.3
[2023-08-05 09:52:06.514848 INFO   ] utils:print_arguments:28 -         use_loss: AAMLoss
[2023-08-05 09:52:06.514882 INFO   ] utils:print_arguments:28 -         use_margin_scheduler: True
[2023-08-05 09:52:06.514915 INFO   ] utils:print_arguments:21 - model_conf:
[2023-08-05 09:52:06.514950 INFO   ] utils:print_arguments:24 -         backbone:
[2023-08-05 09:52:06.514984 INFO   ] utils:print_arguments:26 -                 embd_dim: 192
[2023-08-05 09:52:06.515017 INFO   ] utils:print_arguments:26 -                 pooling_type: ASP
[2023-08-05 09:52:06.515050 INFO   ] utils:print_arguments:24 -         classifier:
[2023-08-05 09:52:06.515084 INFO   ] utils:print_arguments:26 -                 num_blocks: 0
[2023-08-05 09:52:06.515118 INFO   ] utils:print_arguments:21 - optimizer_conf:
[2023-08-05 09:52:06.515154 INFO   ] utils:print_arguments:28 -         learning_rate: 0.001
[2023-08-05 09:52:06.515188 INFO   ] utils:print_arguments:28 -         optimizer: Adam
[2023-08-05 09:52:06.515221 INFO   ] utils:print_arguments:28 -         scheduler: CosineAnnealingLR
[2023-08-05 09:52:06.515254 INFO   ] utils:print_arguments:28 -         scheduler_args: None
[2023-08-05 09:52:06.515289 INFO   ] utils:print_arguments:28 -         weight_decay: 1e-06
[2023-08-05 09:52:06.515323 INFO   ] utils:print_arguments:21 - preprocess_conf:
[2023-08-05 09:52:06.515357 INFO   ] utils:print_arguments:28 -         feature_method: MelSpectrogram
[2023-08-05 09:52:06.515390 INFO   ] utils:print_arguments:24 -         method_args:
[2023-08-05 09:52:06.515426 INFO   ] utils:print_arguments:26 -                 f_max: 14000.0
[2023-08-05 09:52:06.515460 INFO   ] utils:print_arguments:26 -                 f_min: 50.0
[2023-08-05 09:52:06.515493 INFO   ] utils:print_arguments:26 -                 hop_length: 320
[2023-08-05 09:52:06.515527 INFO   ] utils:print_arguments:26 -                 n_fft: 1024
[2023-08-05 09:52:06.515560 INFO   ] utils:print_arguments:26 -                 n_mels: 64
[2023-08-05 09:52:06.515593 INFO   ] utils:print_arguments:26 -                 sample_rate: 16000
[2023-08-05 09:52:06.515626 INFO   ] utils:print_arguments:26 -                 win_length: 1024
[2023-08-05 09:52:06.515660 INFO   ] utils:print_arguments:21 - train_conf:
[2023-08-05 09:52:06.515694 INFO   ] utils:print_arguments:28 -         log_interval: 100
[2023-08-05 09:52:06.515728 INFO   ] utils:print_arguments:28 -         max_epoch: 30
[2023-08-05 09:52:06.515761 INFO   ] utils:print_arguments:30 - use_model: EcapaTdnn
[2023-08-05 09:52:06.515794 INFO   ] utils:print_arguments:31 - ------------------------------------------------
······
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
Sequential                                    [1, 9726]                 --
├─EcapaTdnn: 1-1                              [1, 192]                  --
│    └─Conv1dReluBn: 2-1                      [1, 512, 98]              --
│    │    └─Conv1d: 3-1                       [1, 512, 98]              163,840
│    │    └─BatchNorm1d: 3-2                  [1, 512, 98]              1,024
│    └─Sequential: 2-2                        [1, 512, 98]              --
│    │    └─Conv1dReluBn: 3-3                 [1, 512, 98]              263,168
│    │    └─Res2Conv1dReluBn: 3-4             [1, 512, 98]              86,912
│    │    └─Conv1dReluBn: 3-5                 [1, 512, 98]              263,168
│    │    └─SE_Connect: 3-6                   [1, 512, 98]              262,912
│    └─Sequential: 2-3                        [1, 512, 98]              --
│    │    └─Conv1dReluBn: 3-7                 [1, 512, 98]              263,168
│    │    └─Res2Conv1dReluBn: 3-8             [1, 512, 98]              86,912
│    │    └─Conv1dReluBn: 3-9                 [1, 512, 98]              263,168
│    │    └─SE_Connect: 3-10                  [1, 512, 98]              262,912
│    └─Sequential: 2-4                        [1, 512, 98]              --
│    │    └─Conv1dReluBn: 3-11                [1, 512, 98]              263,168
│    │    └─Res2Conv1dReluBn: 3-12            [1, 512, 98]              86,912
│    │    └─Conv1dReluBn: 3-13                [1, 512, 98]              263,168
│    │    └─SE_Connect: 3-14                  [1, 512, 98]              262,912
│    └─Conv1d: 2-5                            [1, 1536, 98]             2,360,832
│    └─AttentiveStatsPool: 2-6                [1, 3072]                 --
│    │    └─Conv1d: 3-15                      [1, 128, 98]              196,736
│    │    └─Conv1d: 3-16                      [1, 1536, 98]             198,144
│    └─BatchNorm1d: 2-7                       [1, 3072]                 6,144
│    └─Linear: 2-8                            [1, 192]                  590,016
│    └─BatchNorm1d: 2-9                       [1, 192]                  384
├─SpeakerIdentification: 1-2                  [1, 9726]                 1,867,392
===============================================================================================
Total params: 8,012,992
Trainable params: 8,012,992
Non-trainable params: 0
Total mult-adds (M): 468.81
===============================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 10.36
Params size (MB): 32.05
Estimated Total Size (MB): 42.44
===============================================================================================
[2023-08-05 09:52:08.084231 INFO   ] trainer:train:388 - 训练数据：874175
[2023-08-05 09:52:09.186542 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [0/13659], loss: 11.95824, accuracy: 0.00000, learning rate: 0.00100000, speed: 58.09 data/sec, eta: 5 days, 5:24:08
[2023-08-05 09:52:22.477905 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [100/13659], loss: 10.35675, accuracy: 0.00278, learning rate: 0.00100000, speed: 481.65 data/sec, eta: 15:07:15
[2023-08-05 09:52:35.948581 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [200/13659], loss: 10.22089, accuracy: 0.00505, learning rate: 0.00100000, speed: 475.27 data/sec, eta: 15:19:12
[2023-08-05 09:52:49.249098 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [300/13659], loss: 10.00268, accuracy: 0.00706, learning rate: 0.00100000, speed: 481.45 data/sec, eta: 15:07:11
[2023-08-05 09:53:03.716015 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [400/13659], loss: 9.76052, accuracy: 0.00830, learning rate: 0.00100000, speed: 442.74 data/sec, eta: 16:26:16
[2023-08-05 09:53:18.258807 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [500/13659], loss: 9.50189, accuracy: 0.01060, learning rate: 0.00100000, speed: 440.46 data/sec, eta: 16:31:08
[2023-08-05 09:53:31.618354 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [600/13659], loss: 9.26083, accuracy: 0.01256, learning rate: 0.00100000, speed: 479.50 data/sec, eta: 15:10:12
[2023-08-05 09:53:45.439642 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [700/13659], loss: 9.03548, accuracy: 0.01449, learning rate: 0.00099999, speed: 463.63 data/sec, eta: 15:41:08
```

VisualDL：
![VisualDL页面](./docs/images/log.jpg)



# Eval
After training, the prediction model will be saved, and we will use the prediction model to predict the audio features in the test set, and then use the audio features for pairwise comparison to calculate EER and MinDCF.
```shell
python eval.py
```

The output will look like this:
```
······
------------------------------------------------
W0425 08:27:32.057426 17654 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 10.2
W0425 08:27:32.065165 17654 device_context.cc:465] device: 0, cuDNN Version: 7.6.
[2023-03-16 20:20:47.195908 INFO   ] trainer:evaluate:341 - 成功加载模型：models/EcapaTdnn_Fbank/best_model/model.pth
100%|███████████████████████████| 84/84 [00:28<00:00,  2.95it/s]
开始两两对比音频特征...
100%|███████████████████████████| 5332/5332 [00:05<00:00, 1027.83it/s]
评估消耗时间：65s，threshold：0.26，EER: 0.14739, MinDCF: 0.41999
```

# Voiceprint Contrast 

Let's start implementing the voiceprint comparison, create the `infer_contrast.py` program, and write the `infer()` function. When we write the model, the model will have two outputs, the first is the classification output of the model, and the second is the audio feature output. So the output here is the characteristic value of audio, with the characteristic value of audio, you can do voiceprint recognition. We input two voices and get their feature data through the prediction function. Using this feature data, we can find their diagonal cosine value, and the result can be used as their acquaintance degree. This familiarity threshold `threshold` can be modified according to the accuracy requirements of your project.
```shell
python infer_contrast.py --audio_path1=audio/a_1.wav --audio_path2=audio/b_2.wav
```

The output will look like this:
```
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:13 - ----------- 额外配置参数 -----------
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:15 - audio_path1: dataset/a_1.wav
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:15 - audio_path2: dataset/b_2.wav
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:15 - configs: configs/ecapa_tdnn.yml
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:15 - model_path: models/EcapaTdnn_Fbank/best_model/
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:15 - threshold: 0.6
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:15 - use_gpu: True
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:16 - ------------------------------------------------
······································································
W0425 08:29:10.006249 21121 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 10.2
W0425 08:29:10.008555 21121 device_context.cc:465] device: 0, cuDNN Version: 7.6.
成功加载模型参数和优化方法参数：models/EcapaTdnn_Fbank/best_model/model.pth
audio/a_1.wav 和 audio/b_2.wav 不是同一个人，相似度为：-0.09565544128417969
```

# Voiceprint Recognition

Based on the above voiceprint comparison, we create `infer_recognition.py` for voiceprint recognition. We use the same `infer()` prediction function from the voiceprint comparison above to get the speech feature data from both of them. The difference is that the author added `load_audio_db()` and `register()`, and `recognition()`. The first function is to load the speech data in the voiceprint library. These audio are equivalent to registered users, and their registered speech data will be stored here. If a user needs to log in through voiceprint, it is necessary to get the user's voice and the voice in the speech database for voiceprint comparison. If the comparison is successful, it is equivalent to successful login and obtain the information data of the user registration. The second function, `register()`, saves the recording to the voiceprint library and takes the features of the audio and adds them to the data features to be compared. Finally, the `recognition()` function compares the input speech to the speech in the database.
With the function of voiceprint recognition above, readers can complete the voiceprint recognition according to the needs of their own projects. For example, the following is provided by the author to complete the voiceprint recognition through recording. First of all, we must load the speech in the speech library, the speech library folder is `audio_db`, and then record for 3 seconds after the user enters the car, and then the program will automatically record, and use the recorded audio for voiceprint recognition to match the speech in the speech library and obtain the user's information. In this way, the reader can also modify to complete the voiceprint recognition through the service request, for example, provide an API for the APP to call, the user logs in through the voiceprint on the APP, send the recorded voice to the back-end to complete the voiceprint recognition, and then return the result to the APP, provided that the user has registered with the voice, And successfully stored the speech data in the `audio_db` folder.
```shell
python infer_recognition.py
```

The output will look like this:
```
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:13 - ----------- 额外配置参数 -----------
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:15 - audio_db_path: audio_db/
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:15 - configs: configs/ecapa_tdnn.yml
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:15 - model_path: models/EcapaTdnn_Fbank/best_model/
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:15 - record_seconds: 3
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:15 - threshold: 0.6
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:15 - use_gpu: True
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:16 - ------------------------------------------------
······································································
W0425 08:30:13.257884 23889 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 10.2
W0425 08:30:13.260191 23889 device_context.cc:465] device: 0, cuDNN Version: 7.6.
成功加载模型参数和优化方法参数：models/ecapa_tdnn/model.pth
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

# Other Versions
 - Tensorflow：[VoiceprintRecognition-Tensorflow](https://github.com/yeyupiaoling/VoiceprintRecognition-Tensorflow)
 - PaddlePaddle：[VoiceprintRecognition-PaddlePaddle](https://github.com/yeyupiaoling/VoiceprintRecognition-PaddlePaddle)
 - Keras：[VoiceprintRecognition-Keras](https://github.com/yeyupiaoling/VoiceprintRecognition-Keras)


# Reference
1. https://github.com/PaddlePaddle/PaddleSpeech
2. https://github.com/yeyupiaoling/PaddlePaddle-MobileFaceNets
3. https://github.com/yeyupiaoling/PPASR
4. https://github.com/alibaba-damo-academy/3D-Speaker
5. https://github.com/wenet-e2e/wespeaker
