import os
import pickle
import shutil
from io import BufferedReader

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from mvector import SUPPORT_MODEL
from mvector.data_utils.audio import AudioSegment
from mvector.data_utils.featurizer import AudioFeaturizer
from mvector.models.campplus import CAMPPlus
from mvector.models.ecapa_tdnn import EcapaTdnn
from mvector.models.eres2net import ERes2Net
from mvector.models.res2net import Res2Net
from mvector.models.resnet_se import ResNetSE
from mvector.models.tdnn import TDNN
from mvector.utils.logger import setup_logger
from mvector.utils.utils import dict_to_object, print_arguments

logger = setup_logger(__name__)


class MVectorPredictor:
    def __init__(self,
                 configs,
                 threshold=0.6,
                 audio_db_path=None,
                 model_path='models/EcapaTdnn_Fbank/best_model/',
                 use_gpu=True):
        """
        声纹识别预测工具
        :param configs: 配置参数
        :param threshold: 判断是否为同一个人的阈值
        :param audio_db_path: 声纹库路径
        :param model_path: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        # 索引候选数量
        self.cdd_num = 5
        self.threshold = threshold
        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'
        self._audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                 method_args=self.configs.preprocess_conf.get('method_args', {}))
        self._audio_featurizer.to(self.device)
        # 获取模型
        if self.configs.use_model == 'ERes2Net':
            backbone = ERes2Net(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'CAMPPlus':
            backbone = CAMPPlus(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'EcapaTdnn':
            backbone = EcapaTdnn(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'Res2Net':
            backbone = Res2Net(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'ResNetSE':
            backbone = ResNetSE(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'TDNN':
            backbone = TDNN(input_size=self._audio_featurizer.feature_dim, **self.configs.model_conf.backbone)
        else:
            raise Exception(f'{self.configs.use_model} 模型不存在！')
        model = nn.Sequential(backbone)
        model.to(self.device)
        # 加载模型
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'model.pth')
        assert os.path.exists(model_path), f"{model_path} 模型不存在！"
        if torch.cuda.is_available() and use_gpu:
            model_state_dict = torch.load(model_path)
        else:
            model_state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)
        print(f"成功加载模型参数：{model_path}")
        model.eval()
        self.predictor = model

        # 声纹库的声纹特征
        self.audio_feature = None
        # 声纹特征对应的用户名
        self.users_name = []
        # 声纹特征对应的声纹文件路径
        self.users_audio_path = []
        # 加载声纹库
        self.audio_db_path = audio_db_path
        if self.audio_db_path is not None:
            self.audio_indexes_path = os.path.join(audio_db_path, "audio_indexes.bin")
            # 加载声纹库中的声纹
            self.__load_faces(self.audio_db_path)

    # 加载声纹特征索引
    def __load_face_indexes(self):
        # 如果存在声纹特征索引文件就加载
        if not os.path.exists(self.audio_indexes_path): return
        with open(self.audio_indexes_path, "rb") as f:
            indexes = pickle.load(f)
        self.users_name = indexes["users_name"]
        self.audio_feature = indexes["faces_feature"]
        self.users_audio_path = indexes["users_image_path"]

    # 保存声纹特征索引
    def __write_index(self):
        with open(self.audio_indexes_path, "wb") as f:
            pickle.dump({"users_name": self.users_name,
                         "faces_feature": self.audio_feature,
                         "users_image_path": self.users_audio_path}, f)

    # 加载声纹库中的声纹
    def __load_faces(self, audio_db_path):
        # 先加载声纹特征索引
        self.__load_face_indexes()
        os.makedirs(audio_db_path, exist_ok=True)
        audios_path = []
        for name in os.listdir(audio_db_path):
            audio_dir = os.path.join(audio_db_path, name)
            if not os.path.isdir(audio_dir): continue
            for file in os.listdir(audio_dir):
                audios_path.append(os.path.join(audio_dir, file).replace('\\', '/'))
        # 声纹库没数据就跳过
        if len(audios_path) == 0: return
        logger.info('正在加载声纹库数据...')
        input_audios = []
        for audio_path in tqdm(audios_path):
            # 如果声纹特征已经在索引就跳过
            if audio_path in self.users_audio_path: continue
            # 读取声纹库音频
            audio_segment = self._load_audio(audio_path)
            # 获取用户名
            user_name = os.path.basename(os.path.dirname(audio_path))
            self.users_name.append(user_name)
            self.users_audio_path.append(audio_path)
            input_audios.append(audio_segment.samples)
            # 处理一批数据
            if len(input_audios) == self.configs.dataset_conf.eval_conf.batch_size:
                features = self.predict_batch(input_audios)
                if self.audio_feature is None:
                    self.audio_feature = features
                else:
                    self.audio_feature = np.vstack((self.audio_feature, features))
                input_audios = []
        # 处理不满一批的数据
        if len(input_audios) != 0:
            features = self.predict_batch(input_audios)
            if self.audio_feature is None:
                self.audio_feature = features
            else:
                self.audio_feature = np.vstack((self.audio_feature, features))
        assert len(self.audio_feature) == len(self.users_name) == len(self.users_audio_path), '加载的数量对不上！'
        # 将声纹特征保存到索引文件中
        self.__write_index()
        logger.info('声纹库数据加载完成！')

    # 声纹检索
    def __retrieval(self, np_feature):
        labels = []
        for feature in np_feature:
            similarity = cosine_similarity(self.audio_feature, feature[np.newaxis, :]).squeeze()
            abs_similarity = np.abs(similarity)
            # 获取候选索引
            if len(abs_similarity) < self.cdd_num:
                candidate_idx = np.argpartition(abs_similarity, -len(abs_similarity))[-len(abs_similarity):]
            else:
                candidate_idx = np.argpartition(abs_similarity, -self.cdd_num)[-self.cdd_num:]
            # 过滤低于阈值的索引
            remove_idx = np.where(abs_similarity[candidate_idx] < self.threshold)
            candidate_idx = np.delete(candidate_idx, remove_idx)
            # 获取标签最多的值
            candidate_label_list = list(np.array(self.users_name)[candidate_idx])
            candidate_label_dict = {k: v for k, v in zip(candidate_idx, candidate_label_list)}
            if len(candidate_label_list) == 0:
                max_label, score = None, None
            else:
                max_label = max(candidate_label_list, key=candidate_label_list.count)
                scores = [abs_similarity[k] for k, v in candidate_label_dict.items() if v == max_label]
                score = round(sum(scores) / len(scores), 5)
            labels.append([max_label, score])
        return labels

    def _load_audio(self, audio_data, sample_rate=16000):
        """加载音频
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, BufferedReader):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        assert audio_segment.duration >= self.configs.dataset_conf.min_duration, \
            f'音频太短，最小应该为{self.configs.dataset_conf.min_duration}s，当前音频为{audio_segment.duration}s'
        # 重采样
        if audio_segment.sample_rate != self.configs.dataset_conf.sample_rate:
            audio_segment.resample(self.configs.dataset_conf.sample_rate)
        # decibel normalization
        if self.configs.dataset_conf.use_dB_normalization:
            audio_segment.normalize(target_db=self.configs.dataset_conf.target_dB)
        return audio_segment

    def predict(self,
                audio_data,
                sample_rate=16000):
        """预测一个音频的特征

        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整并带格式的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 声纹特征向量
        """
        # 加载音频文件，并进行预处理
        input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        input_data = torch.tensor(input_data.samples, dtype=torch.float32, device=self.device).unsqueeze(0)
        input_len_ratio = torch.tensor([1], dtype=torch.float32, device=self.device)
        audio_feature, _ = self._audio_featurizer(input_data, input_len_ratio)
        # 执行预测
        feature = self.predictor(audio_feature).data.cpu().numpy()[0]
        return feature

    def predict_batch(self, audios_data, sample_rate=16000):
        """预测一批音频的特征

        :param audios_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整并带格式的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 声纹特征向量
        """
        audios_data1 = []
        for audio_data in audios_data:
            # 加载音频文件，并进行预处理
            input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
            audios_data1.append(input_data.samples)
        # 找出音频长度最长的
        batch = sorted(audios_data1, key=lambda a: a.shape[0], reverse=True)
        max_audio_length = batch[0].shape[0]
        batch_size = len(batch)
        # 以最大的长度创建0张量
        inputs = np.zeros((batch_size, max_audio_length), dtype='float32')
        input_lens_ratio = []
        for x in range(batch_size):
            tensor = audios_data1[x]
            seq_length = tensor.shape[0]
            # 将数据插入都0张量中，实现了padding
            inputs[x, :seq_length] = tensor[:]
            input_lens_ratio.append(seq_length / max_audio_length)
        audios_data = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        input_lens_ratio = torch.tensor(input_lens_ratio, dtype=torch.float32, device=self.device)
        audio_feature, _ = self._audio_featurizer(audios_data, input_lens_ratio)
        # 执行预测
        features = self.predictor(audio_feature).data.cpu().numpy()
        return features

    def contrast(self, audio_data1, audio_data2):
        """声纹对比

        param audio_data1: 需要对比的音频1，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        param audio_data2: 需要对比的音频2，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件

        return: 两个音频的相似度
        """
        feature1 = self.predict(audio_data1)
        feature2 = self.predict(audio_data2)
        # 对角余弦值
        dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        return dist

    def register(self,
                 audio_data,
                 user_name: str,
                 sample_rate=16000):
        """声纹注册
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param user_name: 注册用户的名字
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 加载音频文件
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, BufferedReader):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        feature = self.predict(audio_data=audio_segment.samples, sample_rate=audio_segment.sample_rate)
        if self.audio_feature is None:
            self.audio_feature = feature
        else:
            self.audio_feature = np.vstack((self.audio_feature, feature))
        # 保存
        if not os.path.exists(os.path.join(self.audio_db_path, user_name)):
            audio_path = os.path.join(self.audio_db_path, user_name, '0.wav')
        else:
            audio_path = os.path.join(self.audio_db_path, user_name,
                                      f'{len(os.listdir(os.path.join(self.audio_db_path, user_name)))}.wav')
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        audio_segment.to_wav_file(audio_path)
        self.users_audio_path.append(audio_path.replace('\\', '/'))
        self.users_name.append(user_name)
        self.__write_index()
        return True, "注册成功"

    def recognition(self, audio_data, threshold=None, sample_rate=16000):
        """声纹识别
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param threshold: 判断的阈值，如果为None则用创建对象时使用的阈值
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的用户名称，如果为None，即没有识别到用户
        """
        if threshold:
            self.threshold = threshold
        feature = self.predict(audio_data, sample_rate=sample_rate)
        name = self.__retrieval(np_feature=[feature])[0]
        return name

    def get_users(self):
        """获取所有用户

        return: 所有用户
        """
        return self.users_name

    def remove_user(self, user_name):
        """删除用户

        :param user_name: 用户名
        :return:
        """
        if user_name in self.users_name:
            indexes = [i for i in range(len(self.users_name)) if self.users_name[i] == user_name]
            for index in sorted(indexes, reverse=True):
                del self.users_name[index]
                del self.users_audio_path[index]
                self.audio_feature = np.delete(self.audio_feature, index, axis=0)
            self.__write_index()
            shutil.rmtree(os.path.join(self.audio_db_path, user_name))
            return True
        else:
            return False
