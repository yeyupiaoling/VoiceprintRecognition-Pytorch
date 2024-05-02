import torch


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch_sorted = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    freq_size = batch_sorted[0][0].size(1)
    max_freq_length = batch_sorted[0][0].size(0)
    batch_size = len(batch_sorted)
    # 以最大的长度创建0张量
    features = torch.zeros((batch_size, max_freq_length, freq_size), dtype=torch.float32)
    input_lens, labels = [], []
    for x in range(batch_size):
        tensor, label = batch[x]
        seq_length = tensor.size(0)
        # 将数据插入都0张量中，实现了padding
        features[x, :seq_length, :] = tensor[:, :]
        labels.append(label)
        input_lens.append(seq_length)
    labels = torch.tensor(labels, dtype=torch.int64)
    input_lens = torch.tensor(input_lens, dtype=torch.int64)
    return features, labels, input_lens
