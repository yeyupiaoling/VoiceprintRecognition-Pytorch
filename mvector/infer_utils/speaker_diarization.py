# This implementation is adapted from https://github.com/modelscope/modelscope
import numpy as np
import scipy
import sklearn
from sklearn.cluster import k_means
from yeaudio.audio import AudioSegment


class SpeakerDiarization(object):

    def __init__(self):
        self.seg_duration = 1.5
        self.seg_shift = 0.75
        self.sample_rate = 16000
        self.merge_threshold = 0.78
        self.spectral_cluster = SpectralCluster()

    def segments_audio(self, audio_segment: AudioSegment) -> list:
        vad_segments = []
        samples = audio_segment.samples
        self.sample_rate = audio_segment.sample_rate
        vad_time_list = audio_segment.vad(return_seconds=True)
        for t in vad_time_list:
            st = int(t['start'])
            ed = int(t['end'])
            vad_segments.append([st, ed, samples[int(st * self.sample_rate):int(ed * self.sample_rate)]])
        self._check_audio_list(vad_segments)
        segments = self._chunk(vad_segments)
        return segments

    def _check_audio_list(self, audio: list):
        audio_dur = 0
        for i in range(len(audio)):
            seg = audio[i]
            assert seg[1] >= seg[0], 'modelscope error: Wrong time stamps.'
            assert isinstance(seg[2], np.ndarray), 'modelscope error: Wrong data type.'
            assert int(seg[1] * self.sample_rate) - int(
                seg[0] * self.sample_rate
            ) == seg[2].shape[0], 'modelscope error: audio data in list is inconsistent with time length.'
            if i > 0:
                assert seg[0] >= audio[i - 1][1], 'modelscope error: Wrong time stamps.'
            audio_dur += seg[1] - seg[0]
        assert audio_dur > 5, 'modelscope error: The effective audio duration is too short.'

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

    def clustering(self, embeddings: np.ndarray) -> np.ndarray:
        labels = self.spectral_cluster(embeddings)
        labels = self._merge_by_cos(labels, embeddings, self.merge_threshold)
        return labels

    @staticmethod
    def _merge_by_cos(labels, embs, cos_thr):
        # merge the similar speakers by cosine similarity
        assert 0 < cos_thr <= 1
        while True:
            spk_num = labels.max() + 1
            if spk_num == 1:
                break
            spk_center = []
            for i in range(spk_num):
                spk_emb = embs[labels == i].mean(0)
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
        assert len(segments) == len(labels)
        labels = self._correct_labels(labels)
        distribute_res = []
        for i in range(len(segments)):
            distribute_res.append([segments[i][0], segments[i][1], labels[i]])
        # merge the same speakers chronologically
        distribute_res = self._merge_seque(distribute_res)

        def is_overlapped(t1, t2):
            if t1 > t2 + 1e-4:
                return True
            return False

        # distribute the overlap region
        for i in range(1, len(distribute_res)):
            if is_overlapped(distribute_res[i - 1][1], distribute_res[i][0]):
                p = (distribute_res[i][0] + distribute_res[i - 1][1]) / 2
                distribute_res[i][0] = p
                distribute_res[i - 1][1] = p

        # smooth the result
        distribute_res = self._smooth(distribute_res)

        results = []
        for result in distribute_res:
            results.append(dict(speaker=result[2], start=round(result[0], 3), end=round(result[1], 3)))

        return results

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

    @staticmethod
    def _merge_seque(distribute_res):
        res = [distribute_res[0]]
        for i in range(1, len(distribute_res)):
            if distribute_res[i][2] != res[-1][2] or distribute_res[i][0] > res[-1][1]:
                res.append(distribute_res[i])
            else:
                res[-1][1] = distribute_res[i][1]
        return res

    def _smooth(self, res, min_duration=1):
        # short segments are assigned to nearest speakers.
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
        # merge the speakers
        res = self._merge_seque(res)

        return res


class SpectralCluster:
    r"""A spectral clustering mehtod using unnormalized Laplacian of affinity matrix.
    This implementation is adapted from https://github.com/speechbrain/speechbrain.
    """

    def __init__(self, min_num_spks=1, max_num_spks=15, pval=0.022):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.pval = pval

    def __call__(self, X, oracle_num=None):
        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)

        # Refining similarity matrix with pval
        prunned_sim_mat = self.p_pruning(sim_mat)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_prund_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num)

        # Perform clustering
        labels = self.cluster_embs(emb, num_of_spk)

        return labels

    def get_sim_mat(self, X):
        # Cosine similarities
        M = sklearn.metrics.pairwise.cosine_similarity(X, X)
        return M

    def p_pruning(self, A):
        if A.shape[0] * self.pval < 6:
            pval = 6. / A.shape[0]
        else:
            pval = self.pval

        n_elems = int((1 - pval) * A.shape[0])

        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0
        return A

    @staticmethod
    def get_laplacian(M):
        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    def get_spec_embs(self, L, k_oracle=None):
        lambdas, eig_vecs = scipy.linalg.eigh(L)

        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(
                lambdas[self.min_num_spks - 1:self.max_num_spks + 1])
            num_of_spk = np.argmax(lambda_gap_list) + self.min_num_spks

        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    @staticmethod
    def cluster_embs(emb, k):
        _, labels, _ = k_means(emb, k, n_init='auto')
        return labels

    def getEigenGaps(self, eig_vals):
        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            eig_vals_gap_list.append(gap)
        return eig_vals_gap_list
