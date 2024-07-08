from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class PKSampler(Sampler):
    """随机取一批数据，保证每个类别的数量都是相同的。

    Args:
        dataset (Dataset): 数据的Dataset
        batch_size (int): batch size
        sample_per_id (int): 每个类别的样本数量
        shuffle (bool, optional): 是否随机打乱数据
        drop_last (bool, optional): 是否丢掉最后一个batch
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 sample_per_id,
                 shuffle=True,
                 drop_last=True):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        assert batch_size % sample_per_id == 0, f"batch_size({batch_size})必须是sample_per_id({sample_per_id})的整数倍"
        self.sample_per_id = sample_per_id
        self.label_dict = defaultdict(list)
        for idx, label in enumerate(self.dataset.labels):
            self.label_dict[label].append(idx)
        self.label_list = list(self.label_dict)
        assert len(self.label_list) * self.sample_per_id >= self.batch_size, \
            f"batch_size({self.batch_size})必须大于等于label_list({len(self.label_list)})*sample_per_id({self.sample_per_id})"
        self.epoch = 0
        self.prob_list = np.array([1 / len(self.label_list)] * len(self.label_list))
        diff = np.abs(sum(self.prob_list) - 1)
        if diff > 0.00000001:
            self.prob_list[-1] = 1 - sum(self.prob_list[:-1])

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(self.label_list)
            np.random.RandomState(self.epoch).shuffle(self.prob_list)
            self.epoch += 1

        label_per_batch = self.batch_size // self.sample_per_id
        for _ in range(len(self)):
            batch_index = []
            batch_label_list = np.random.choice(self.label_list, size=label_per_batch, replace=False, p=self.prob_list)
            for label_i in batch_label_list:
                label_i_indexes = self.label_dict[label_i]
                batch_index.extend(
                    np.random.choice(label_i_indexes, size=self.sample_per_id,
                                     replace=not self.sample_per_id <= len(label_i_indexes)))
            # 再次随机打乱
            if self.shuffle:
                np.random.shuffle(batch_index)
            if not self.drop_last or len(batch_index) == self.batch_size:
                yield batch_index
        self.epoch += 1
