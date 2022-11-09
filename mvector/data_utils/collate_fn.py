import numpy as np
import torch


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)
    freq_size = batch[0][0].shape[1]
    max_audio_length = batch[0][0].shape[0]
    batch_size = len(batch)
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, max_audio_length, freq_size), dtype='float32')
    labels = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        labels.append(sample[1])
        seq_length = tensor.shape[0]
        # 将数据插入都0张量中，实现了padding
        inputs[x, :seq_length, :] = tensor[:, :]
    labels = np.array(labels, dtype='int64')
    # 打乱数据
    return torch.tensor(inputs), torch.tensor(labels)
