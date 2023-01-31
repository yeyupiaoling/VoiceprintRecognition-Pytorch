import numpy as np
import torch


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)
    max_audio_length = batch[0][0].shape[0]
    batch_size = len(batch)
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, max_audio_length), dtype='float32')
    input_lens_ratio = []
    labels = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        labels.append(sample[1])
        seq_length = tensor.shape[0]
        # 将数据插入都0张量中，实现了padding
        inputs[x, :seq_length] = tensor[:]
        input_lens_ratio.append(seq_length/max_audio_length)
    input_lens_ratio = np.array(input_lens_ratio, dtype='float32')
    labels = np.array(labels, dtype='int64')
    return torch.tensor(inputs), torch.tensor(labels), torch.tensor(input_lens_ratio)
