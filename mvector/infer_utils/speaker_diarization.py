# This implementation is adapted from https://github.com/modelscope/modelscope
import numpy as np
import scipy
import sklearn
from sklearn.cluster import k_means
from yeaudio.audio import AudioSegment


class SpeakerDiarization(object):

    def __init__(self, seg_duration=1.5, seg_shift=0.75, sample_rate=16000, merge_threshold=0.78):
        """说话人日志工具

        Args:
            seg_duration (float, optional): 每个分割片段的持续时间（秒），默认为1.5秒。
            seg_shift (float, optional): 分割片段之间的时间间隔（秒），默认为0.75秒。
            sample_rate (int, optional): 音频采样率，默认为16000Hz。
            merge_threshold (float, optional): 合并片段的阈值，默认为0.78。当两个片段之间的相似度大于此阈值时，将合并这两个片段。
        """
        self.seg_duration = seg_duration
        self.seg_shift = seg_shift
        self.sample_rate = sample_rate
        self.merge_threshold = merge_threshold
        self.spectral_cluster = SpectralCluster()

    def segments_audio(self, audio_segment: AudioSegment) -> list:
        """ 从音频段中分割出有效的语音段。

        Args:
            audio_segment (AudioSegment): 要分割的音频段对象。
        Returns:
            list: 分割出的有效语音段列表，每个元素是一个包含起始时间戳、结束时间戳和对应音频样本的列表。
        """
        vad_segments = []
        samples = audio_segment.samples
        self.sample_rate = audio_segment.sample_rate
        vad_time_list = audio_segment.vad(return_seconds=True)
        for t in vad_time_list:
            st = round(t['start'], 3)
            ed = round(t['end'], 3)
            vad_segments.append([st, ed, samples[int(st * self.sample_rate):int(ed * self.sample_rate)]])
        self._check_audio_list(vad_segments)
        segments = self._chunk(vad_segments)
        return segments

    # 检查分割的结果数据是否符合要求
    def _check_audio_list(self, audio: list):
        audio_duration = 0
        for i in range(len(audio)):
            seg = audio[i]
            assert seg[1] >= seg[0], '分割的时间戳错误'
            assert isinstance(seg[2], np.ndarray), '数据的类型不正确'
            assert int(seg[1] * self.sample_rate) - int(seg[0] * self.sample_rate) == seg[2].shape[0], '时间长度和数据长度不匹配'
            if i > 0:
                assert seg[0] >= audio[i - 1][1], 'modelscope error: Wrong time stamps.'
            audio_duration += seg[1] - seg[0]
        assert audio_duration > 5, f'音频时间过段，应当大于5秒，当前长度是{audio_duration}秒'

    # 将音频片段继续细分割成固定长度的片段
    def _chunk(self, vad_segments: list) -> list:

        def seg_chunk(seg_data):
            seg_st = seg_data[0]
            data = seg_data[2]
            chunk_len = int(self.seg_duration * self.sample_rate)
            chunk_shift = int(self.seg_shift * self.sample_rate)
            last_chunk_ed = 0
            seg_res = []
            for chunk_st in range(0, data.shape[0], chunk_shift):
                chunk_ed = min(chunk_st + chunk_len, data.shape[0])
                if chunk_ed <= last_chunk_ed:
                    break
                last_chunk_ed = chunk_ed
                chunk_st = max(0, chunk_ed - chunk_len)
                chunk_data = data[chunk_st:chunk_ed]
                if chunk_data.shape[0] < chunk_len:
                    chunk_data = np.pad(chunk_data, (0, chunk_len - chunk_data.shape[0]), 'constant')
                seg_res.append([
                    chunk_st / self.sample_rate + seg_st, chunk_ed / self.sample_rate + seg_st,
                    chunk_data
                ])
            return seg_res

        segs = []
        for i, s in enumerate(vad_segments):
            segs.extend(seg_chunk(s))
        return segs

    def clustering(self, embeddings: np.ndarray, speaker_num=None) -> [np.ndarray, np.ndarray]:
        """聚类音频特征向量，返回聚类后的标签数组

        Args:
            embeddings (np.ndarray): 音频特征向量数组，形状为 (n_samples, embedding_dim)
            speaker_num (int): 说话人数量，提供说话人数量可以提高准确率
        Returns:
            Dict[np.ndarray, dict]: 聚类后的标签数组，形状为 (n_samples,)
        """
        labels = self.spectral_cluster(embeddings, oracle_num=speaker_num)
        labels = self._correct_labels(labels)
        # 每个说话人特征向量平均值
        spk_num = labels.max() + 1
        spk_center = []
        for i in range(spk_num):
            spk_emb = embeddings[labels == i].mean(0)
            spk_center.append(spk_emb)
        assert len(spk_center) > 0
        spk_center_embeddings = np.stack(spk_center, axis=0)
        labels = self._merge_by_cos(labels, spk_center, self.merge_threshold)
        return labels, spk_center_embeddings

    # 通过余弦相似度合并相似说话人
    @staticmethod
    def _merge_by_cos(labels, spk_center_emb, cos_thr):
        assert 0 < cos_thr <= 1
        while True:
            spk_num = labels.max() + 1
            if spk_num == 1:
                break
            spk_center = []
            for i in range(spk_num):
                spk_emb = spk_center_emb[i]
                spk_center.append(spk_emb)
            assert len(spk_center) > 0
            spk_center = np.stack(spk_center, axis=0)
            norm_spk_center = spk_center / np.linalg.norm(spk_center, axis=1, keepdims=True)
            affinity = np.matmul(norm_spk_center, norm_spk_center.T)
            affinity = np.triu(affinity, 1)
            spks = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[spks] < cos_thr:
                break
            for i in range(len(labels)):
                if labels[i] == spks[1]:
                    labels[i] = spks[0]
                elif labels[i] > spks[1]:
                    labels[i] -= 1
        return labels

    def postprocess(self, segments: list, labels: np.ndarray) -> list:
        """对音频分割结果进行后处理，包括标签校正、片段合并、重叠区域分配和平滑处理。

        Args:
            segments (list): 包含分割的数据列表，每个元素是一个包含起始时间、结束时间，音频数据。
            labels (np.ndarray): 包含每个音频片段对应说话人标签的数组。
        Returns:
            list: 包含处理后的音频片段信息的列表，包含说话人标签、起始时间和结束时间。
        """
        assert len(segments) == len(labels)
        distribute_res = []
        for i in range(len(segments)):
            distribute_res.append([segments[i][0], segments[i][1], labels[i]])
        # 按时间顺序合并相同的说话人
        distribute_res = self._merge_seque(distribute_res)

        def is_overlapped(t1, t2):
            if t1 > t2 + 1e-4:
                return True
            return False

        # 分割重叠区域
        for i in range(1, len(distribute_res)):
            if is_overlapped(distribute_res[i - 1][1], distribute_res[i][0]):
                p = (distribute_res[i][0] + distribute_res[i - 1][1]) / 2
                distribute_res[i][0] = p
                distribute_res[i - 1][1] = p

        # 平滑处理
        distribute_res = self._smooth(distribute_res)

        # 将结果转换为字典形式
        results = []
        for result in distribute_res:
            results.append(dict(speaker=result[2], start=round(result[0], 3), end=round(result[1], 3)))

        return results

    # 重排序标签
    @staticmethod
    def _correct_labels(labels):
        labels_id = 0
        id2id = {}
        new_labels = []
        for i in labels:
            if i not in id2id:
                id2id[i] = labels_id
                labels_id += 1
            new_labels.append(id2id[i])
        return np.array(new_labels)

    # 合并连续且属于同一说话人的音频片段
    @staticmethod
    def _merge_seque(distribute_res):
        res = [distribute_res[0]]
        for i in range(1, len(distribute_res)):
            if distribute_res[i][2] != res[-1][2] or distribute_res[i][0] > res[-1][1]:
                res.append(distribute_res[i])
            else:
                res[-1][1] = distribute_res[i][1]
        return res

    # 对结果进行平滑处理，主要是处理时间长度过短的片段
    def _smooth(self, res, min_duration=1):
        for i in range(len(res)):
            res[i][0] = round(res[i][0], 2)
            res[i][1] = round(res[i][1], 2)
            if res[i][1] - res[i][0] < min_duration:
                if i == 0:
                    res[i][2] = res[i + 1][2]
                elif i == len(res) - 1:
                    res[i][2] = res[i - 1][2]
                elif res[i][0] - res[i - 1][1] <= res[i + 1][0] - res[i][1]:
                    res[i][2] = res[i - 1][2]
                else:
                    res[i][2] = res[i + 1][2]
        # 合并说话人
        res = self._merge_seque(res)
        return res


class SpectralCluster:
    def __init__(self, min_num_spks=1, max_num_spks=15, pval=0.022):
        """实现了基于相似度矩阵的非归一化拉普拉斯矩阵的谱聚类方法。

        :param min_num_spks: 聚类的最小数量，默认为1。
        :type min_num_spks: int
        :param max_num_spks: 聚类的最大数量，默认为15。
        :type max_num_spks: int
        :param pval: 用于相似度矩阵修剪的阈值，默认为0.022。
        :type pval: float
        """
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.pval = pval

    # 对输入数据X进行谱聚类，返回聚类标签
    def __call__(self, X, oracle_num=None):
        """

        :param X: 输入数据，形状为[n_samples, n_features]
        :type X: np.ndarray
        :param oracle_num: 聚类数量，默认为None，此时将根据特征间隙自动选择聚类数量。
        :type oracle_num: int
        :return: 聚类标签，形状为[n_samples]
        """
        sim_mat = self.get_sim_mat(X)
        prunned_sim_mat = self.p_pruning(sim_mat)
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)
        laplacian = self.get_laplacian(sym_prund_sim_mat)
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num)
        labels = self.cluster_embs(emb, num_of_spk)
        return labels

    # 计算输入数据X的相似度矩阵
    @staticmethod
    def get_sim_mat(X):
        # Cosine similarities
        M = sklearn.metrics.pairwise.cosine_similarity(X, X)
        return M

    # 根据阈值pval修剪相似度矩阵A
    def p_pruning(self, A):
        if A.shape[0] * self.pval < 6:
            pval = 6. / A.shape[0]
        else:
            pval = self.pval
        n_elems = int((1 - pval) * A.shape[0])

        # 关联矩阵中的每一行中的前n_elems个最小值下标
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]
            # 用0替换较小的相似度值
            A[i, low_indexes] = 0
        return A

    # 计算对称相似度矩阵M的拉普拉斯矩阵
    @staticmethod
    def get_laplacian(M):
        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    # 计算拉普拉斯矩阵L的谱嵌入，并根据特征间隙或指定的oracle_num确定聚类数量
    def get_spec_embs(self, L, k_oracle=None):
        lambdas, eig_vecs = scipy.linalg.eigh(L)

        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.get_eigen_gaps(lambdas[self.min_num_spks - 1:self.max_num_spks + 1])
            num_of_spk = np.argmax(lambda_gap_list) + self.min_num_spks

        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    # 使用k-means算法对谱嵌入emb进行聚类，返回聚类标签
    @staticmethod
    def cluster_embs(emb, k):
        _, labels, _ = k_means(emb, k, n_init="auto")
        return labels

    # 计算特征值的间隙列表
    @staticmethod
    def get_eigen_gaps(eig_vals):
        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            eig_vals_gap_list.append(gap)
        return eig_vals_gap_list
