import json
import os
import platform
import shutil
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from tqdm import tqdm
from visualdl import LogWriter

from mvector import SUPPORT_MODEL, __version__
from mvector.data_utils.collate_fn import collate_fn
from mvector.data_utils.featurizer import AudioFeaturizer
from mvector.data_utils.reader import CustomDataset
from mvector.data_utils.spec_aug import SpecAug
from mvector.metric.metrics import compute_fnr_fpr, compute_eer, compute_dcf, accuracy
from mvector.models.campplus import CAMPPlus
from mvector.models.ecapa_tdnn import EcapaTdnn
from mvector.models.eres2net import ERes2Net
from mvector.models.fc import SpeakerIdentification
from mvector.models.loss import AAMLoss, CELoss, AMLoss, ARMLoss, SubCenterLoss, SphereFace2
from mvector.models.res2net import Res2Net
from mvector.models.resnet_se import ResNetSE
from mvector.models.tdnn import TDNN
from mvector.utils.logger import setup_logger
from mvector.utils.scheduler import WarmupCosineSchedulerLR, MarginScheduler
from mvector.utils.utils import dict_to_object, print_arguments

logger = setup_logger(__name__)


class MVectorTrainer(object):
    def __init__(self, configs, use_gpu=True):
        """ mvector集成工具类

        :param configs: 配置字典
        :param use_gpu: 是否使用GPU训练模型
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        self.use_gpu = use_gpu
        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'
        self.model = None
        self.backbone = None
        self.model_output_name = '1.weight'
        self.train_method = None
        self.original_weight = {}
        self.old_num_class = None
        self.enroll_loader = None
        self.trials_loader = None
        self.margin_scheduler = None
        self.amp_scaler = None
        # 获取特征器
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))
        self.audio_featurizer.to(self.device)
        # 特征增强
        self.spec_aug = SpecAug(**self.configs.dataset_conf.get('spec_aug_args', {}))
        self.spec_aug.to(self.device)

        if platform.system().lower() == 'windows':
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('Windows系统不支持多线程读取数据，已自动关闭！')

    def __setup_dataloader(self, is_train=False):
        if is_train:
            self.train_dataset = CustomDataset(data_list_path=self.configs.dataset_conf.train_list,
                                               do_vad=self.configs.dataset_conf.do_vad,
                                               max_duration=self.configs.dataset_conf.max_duration,
                                               min_duration=self.configs.dataset_conf.min_duration,
                                               sample_rate=self.configs.dataset_conf.sample_rate,
                                               aug_conf=self.configs.dataset_conf.aug_conf,
                                               num_speakers=self.configs.model_conf.classifier.num_speakers,
                                               use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                               target_dB=self.configs.dataset_conf.target_dB,
                                               mode='train')
            train_sampler = None
            if torch.cuda.device_count() > 1:
                # 设置支持多卡训练
                train_sampler = DistributedSampler(dataset=self.train_dataset)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           collate_fn=collate_fn,
                                           shuffle=(train_sampler is None),
                                           sampler=train_sampler,
                                           **self.configs.dataset_conf.dataLoader)
        # 获取评估的注册数据和检验数据
        self.enroll_dataset = CustomDataset(data_list_path=self.configs.dataset_conf.enroll_list,
                                            do_vad=self.configs.dataset_conf.do_vad,
                                            max_duration=self.configs.dataset_conf.eval_conf.max_duration,
                                            min_duration=self.configs.dataset_conf.min_duration,
                                            sample_rate=self.configs.dataset_conf.sample_rate,
                                            use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                            target_dB=self.configs.dataset_conf.target_dB,
                                            mode='eval')
        self.enroll_loader = DataLoader(dataset=self.enroll_dataset,
                                        collate_fn=collate_fn,
                                        batch_size=self.configs.dataset_conf.eval_conf.batch_size,
                                        num_workers=self.configs.dataset_conf.dataLoader.num_workers)
        self.trials_dataset = CustomDataset(data_list_path=self.configs.dataset_conf.trials_list,
                                            do_vad=self.configs.dataset_conf.do_vad,
                                            max_duration=self.configs.dataset_conf.eval_conf.max_duration,
                                            min_duration=self.configs.dataset_conf.min_duration,
                                            sample_rate=self.configs.dataset_conf.sample_rate,
                                            use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                            target_dB=self.configs.dataset_conf.target_dB,
                                            mode='eval')
        self.trials_loader = DataLoader(dataset=self.trials_dataset,
                                        collate_fn=collate_fn,
                                        batch_size=self.configs.dataset_conf.eval_conf.batch_size,
                                        num_workers=self.configs.dataset_conf.dataLoader.num_workers)

    def __setup_model(self, input_size, is_train=False):
        # 获取模型
        if self.configs.use_model == 'ERes2Net':
            self.backbone = ERes2Net(input_size=input_size, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'CAMPPlus':
            self.backbone = CAMPPlus(input_size=input_size, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'EcapaTdnn':
            self.backbone = EcapaTdnn(input_size=input_size, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'Res2Net':
            self.backbone = Res2Net(input_size=input_size, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'ResNetSE':
            self.backbone = ResNetSE(input_size=input_size, **self.configs.model_conf.backbone)
        elif self.configs.use_model == 'TDNN':
            self.backbone = TDNN(input_size=input_size, **self.configs.model_conf.backbone)
        else:
            raise Exception(f'{self.configs.use_model} 模型不存在！')

        # 获取训练所需的函数
        if is_train:
            if self.configs.train_conf.enable_amp:
                self.amp_scaler = torch.cuda.amp.GradScaler(init_scale=1024)
            use_loss = self.configs.loss_conf.get('use_loss', 'AAMLoss')
            # 获取分类器
            num_class = self.configs.model_conf.classifier.num_speakers
            # 语速扰动要增加分类数量
            if self.configs.dataset_conf.aug_conf.speed_perturb:
                if self.train_method is not None:
                    assert not self.configs.dataset_conf.aug_conf.speed_perturb_3_class, \
                        '增量学习不能使用语速扰动增加分类数量'
                if self.configs.dataset_conf.aug_conf.speed_perturb_3_class:
                    self.configs.model_conf.classifier.num_speakers = num_class * 3
            # 分类器
            classifier = SpeakerIdentification(input_dim=self.backbone.embd_dim,
                                               loss_type=use_loss,
                                               **self.configs.model_conf.classifier)
            # 合并模型
            self.model = nn.Sequential(self.backbone, classifier)
            # print(self.model)
            # 获取损失函数
            loss_args = self.configs.loss_conf.get('args', {})
            loss_args = loss_args if loss_args is not None else {}
            if use_loss == 'AAMLoss':
                self.loss = AAMLoss(**loss_args)
            elif use_loss == 'SphereFace2':
                self.loss = SphereFace2(**loss_args)
            elif use_loss == 'SubCenterLoss':
                self.loss = SubCenterLoss(**loss_args)
            elif use_loss == 'AMLoss':
                self.loss = AMLoss(**loss_args)
            elif use_loss == 'ARMLoss':
                self.loss = ARMLoss(**loss_args)
            elif use_loss == 'CELoss':
                self.loss = CELoss(**loss_args)
            else:
                raise Exception(f'没有{use_loss}损失函数！')
            # 损失函数margin调度器
            if self.configs.loss_conf.get('use_margin_scheduler', False):
                margin_scheduler_args = dict(increase_start_epoch=int(self.configs.train_conf.max_epoch * 0.3),
                                             fix_epoch=int(self.configs.train_conf.max_epoch * 0.7),
                                             initial_margin=0.0,
                                             final_margin=0.3)
                if self.configs.loss_conf.margin_scheduler_args:
                    for k, v in self.configs.loss_conf.margin_scheduler_args.items():
                        margin_scheduler_args[k] = v
                self.margin_scheduler = MarginScheduler(criterion=self.loss,
                                                        step_per_epoch=len(self.train_loader),
                                                        **margin_scheduler_args)
            # 获取优化方法
            optimizer = self.configs.optimizer_conf.optimizer
            if optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                                  lr=self.configs.optimizer_conf.learning_rate,
                                                  weight_decay=self.configs.optimizer_conf.weight_decay)
            elif optimizer == 'AdamW':
                self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                                   lr=self.configs.optimizer_conf.learning_rate,
                                                   weight_decay=self.configs.optimizer_conf.weight_decay)
            elif optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                                 momentum=self.configs.optimizer_conf.get('momentum', 0.9),
                                                 lr=self.configs.optimizer_conf.learning_rate,
                                                 weight_decay=self.configs.optimizer_conf.weight_decay)
            else:
                raise Exception(f'不支持优化方法：{optimizer}')
            # 学习率衰减函数
            scheduler_args = self.configs.optimizer_conf.get('scheduler_args', {}) \
                if self.configs.optimizer_conf.get('scheduler_args', {}) is not None else {}
            if self.configs.optimizer_conf.scheduler == 'CosineAnnealingLR':
                max_step = int(self.configs.train_conf.max_epoch * 1.2) * len(self.train_loader)
                self.scheduler = CosineAnnealingLR(optimizer=self.optimizer,
                                                   T_max=max_step,
                                                   **scheduler_args)
            elif self.configs.optimizer_conf.scheduler == 'WarmupCosineSchedulerLR':
                self.scheduler = WarmupCosineSchedulerLR(optimizer=self.optimizer,
                                                         fix_epoch=self.configs.train_conf.max_epoch,
                                                         step_per_epoch=len(self.train_loader),
                                                         **scheduler_args)
            else:
                raise Exception(f'不支持学习率衰减函数：{self.configs.optimizer_conf.scheduler}')
        else:
            # 不训练模型不包含分类器
            self.model = nn.Sequential(self.backbone)
            self.model.to(self.device)
        self.model.to(self.device)
        summary(self.model, (1, 98, self.audio_featurizer.feature_dim))
        # 使用Pytorch2.0的编译器
        if self.configs.train_conf.use_compile and torch.__version__ >= "2" and platform.system().lower() != 'windows':
            self.model = torch.compile(self.model, mode="reduce-overhead")

    # 改变分类器输出权重大小
    def __change_model_output_size(self, model_state_dict, hp_alpha=0.2):
        if self.model_output_name not in model_state_dict.keys():
            return model_state_dict
        self.old_num_class = model_state_dict[self.model_output_name].size(0)
        new_num_class = self.configs.model_conf.classifier.num_speakers
        logger.info(f'【增量学习】改变分类器输出权重大小，原分类器输出大小：{self.old_num_class}, 新分类器输出大小：{new_num_class}')

        output = model_state_dict[self.model_output_name]
        position_ids = torch.LongTensor(list(range(new_num_class)))
        i = position_ids // self.old_num_class
        j = position_ids % self.old_num_class
        base_output = (output - output[0:1] * hp_alpha) / (1 - hp_alpha)

        position_output = hp_alpha * base_output[i] + (1 - hp_alpha) * base_output[j]
        model_state_dict[self.model_output_name] = position_output
        return model_state_dict

    # 加载预训练模型
    def __load_pretrained(self, pretrained_model):
        if self.train_method is not None:
            assert pretrained_model is not None, '增量学习必须要设置预训练模型!'
        if pretrained_model is not None:
            if os.path.isdir(pretrained_model):
                pretrained_model = os.path.join(pretrained_model, 'model.pth')
            assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"
            model_state_dict = torch.load(pretrained_model)
            # 增量学习改变分类器输出权重大小
            if self.train_method is not None:
                model_state_dict = self.__change_model_output_size(model_state_dict)
                # 获取原始的模型参数，用于计算增量学习损失
                for param_name, param in model_state_dict.items():
                    self.original_weight[param_name] = param.data.clone()
            else:
                for param_name, param in self.model.named_parameters():
                    if param_name in model_state_dict.keys() and model_state_dict[param_name].size() != param.size():
                        logger.warning(f'模型参数{param_name}的大小不匹配，权重的是{model_state_dict[param_name].size()}, '
                                       f'模型的是{param.size()}，已忽略！')
                        del model_state_dict[param_name]
            # 加载权重
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                missing_keys, unexpected_keys = self.model.module.load_state_dict(model_state_dict, strict=False)
            else:
                missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
            if len(unexpected_keys) > 0:
                logger.warning('Unexpected key(s) in state_dict: {}. '
                               .format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                logger.warning('Missing key(s) in state_dict: {}. '
                               .format(', '.join('"{}"'.format(k) for k in missing_keys)))
            logger.info('成功加载预训练模型：{}'.format(pretrained_model))

    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        best_eer = 1
        last_model_dir = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pth'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pth'))):
            # 自动获取最新保存的模型
            if resume_model is None: resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pth')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pth')), "优化方法参数文件不存在！"
            state_dict = torch.load(os.path.join(resume_model, 'model.pth'))
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(torch.load(os.path.join(resume_model, 'optimizer.pth')))
            # 自动混合精度参数
            if self.amp_scaler is not None and os.path.exists(os.path.join(resume_model, 'scaler.pth')):
                self.amp_scaler.load_state_dict(torch.load(os.path.join(resume_model, 'scaler.pth')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                if 'eer' in json_data.keys():
                    best_eer = json_data['eer']
            logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
            self.optimizer.step()
            # 恢复学习率和margin
            if last_epoch >= 0:
                [self.scheduler.step() for _ in range((last_epoch + 1) * len(self.train_loader))]
                if self.margin_scheduler:
                    self.margin_scheduler.step((last_epoch + 1) * len(self.train_loader))
        return last_epoch, best_eer

    # 保存模型
    def __save_checkpoint(self, save_model_path, epoch_id, eer=None, min_dcf=None, threshold=None, best_model=False):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        if best_model:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'best_model')
        else:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'epoch_{}'.format(epoch_id))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
        torch.save(state_dict, os.path.join(model_path, 'model.pth'))
        # 自动混合精度参数
        if self.amp_scaler is not None:
            torch.save(self.amp_scaler.state_dict(), os.path.join(model_path, 'scaler.pth'))
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            data = {"last_epoch": epoch_id, "version": __version__}
            if eer is not None:
                data['threshold'] = threshold
                data['eer'] = eer
                data['min_dcf'] = min_dcf
            if self.margin_scheduler:
                data['margin'] = self.margin_scheduler.get_margin()
            f.write(json.dumps(data, ensure_ascii=False))
        if not best_model:
            last_model_path = os.path.join(save_model_path,
                                           f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                           'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型
            old_model_path = os.path.join(save_model_path,
                                          f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                          'epoch_{}'.format(epoch_id - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        logger.info('已保存模型：{}'.format(model_path))

    def __train_epoch(self, epoch_id, save_model_path, local_rank, writer, nranks=0, il_ratio=10.0):
        train_times, accuracies, loss_sum, il_losses = [], [], [], []
        start = time.time()
        use_loss = self.configs.loss_conf.get('use_loss', 'AAMLoss')
        sum_batch = len(self.train_loader) * self.configs.train_conf.max_epoch
        for batch_id, (audio, label, input_lens_ratio) in enumerate(self.train_loader):
            if nranks > 1:
                audio = audio.to(local_rank)
                input_lens_ratio = input_lens_ratio.to(local_rank)
                label = label.to(local_rank).long()
            else:
                audio = audio.to(self.device)
                input_lens_ratio = input_lens_ratio.to(self.device)
                label = label.to(self.device).long()
            features, _ = self.audio_featurizer(audio, input_lens_ratio)
            # 特征增强
            if self.configs.dataset_conf.use_spec_aug:
                features = self.spec_aug(features)
            # 执行模型计算，是否开启自动混合精度
            with torch.cuda.amp.autocast(enabled=self.configs.train_conf.enable_amp):
                output = self.model(features)
            # 计算损失值
            los = self.loss(output, label)
            # 增量学习损失
            if self.train_method == 'ewc':
                losses = []
                for n, p in self.model.named_parameters():
                    # 每个参数都有mean和fisher
                    mean = self.original_weight[n.replace("module.", "")]
                    if self.model_output_name == n:
                        pass
                        # losses.append(((p - mean)[:self.old_num_class, :] ** 2).sum())
                    else:
                        losses.append(((p - mean) ** 2).sum())
                ewc_loss = sum(losses)
                los = los + (ewc_loss * il_ratio)
                il_losses.append(ewc_loss * il_ratio)
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                # loss缩放，乘以系数loss_scaling
                scaled = self.amp_scaler.scale(los)
                scaled.backward()
            else:
                los.backward()
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                self.amp_scaler.unscale_(self.optimizer)
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

            # 计算准确率
            if use_loss == 'SubCenterLoss':
                loss_args = self.configs.loss_conf.get('args', {})
                cosine = torch.reshape(output, (-1, output.shape[1] // loss_args.K, loss_args.K))
                output, _ = torch.max(cosine, 2)
            acc = accuracy(output, label)
            accuracies.append(acc)
            loss_sum.append(los.data.cpu().numpy())
            train_times.append((time.time() - start) * 1000)

            # 多卡训练只使用一个进程打印
            if batch_id % self.configs.train_conf.log_interval == 0 and local_rank == 0:
                # 计算每秒训练数据量
                train_speed = self.configs.dataset_conf.dataLoader.batch_size / (
                        sum(train_times) / len(train_times) / 1000)
                # 计算剩余时间
                eta_sec = (sum(train_times) / len(train_times)) * (
                        sum_batch - (epoch_id - 1) * len(self.train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                margin_str = f'margin: {self.margin_scheduler.get_margin()}' if self.margin_scheduler else ''
                il_loss_str = f'il_loss: {sum(il_losses) / len(il_losses):.5f}, ' if self.train_method is not None else ''
                logger.info(f'Train epoch: [{epoch_id}/{self.configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {sum(loss_sum) / len(loss_sum):.5f}, {il_loss_str}'
                            f'accuracy: {sum(accuracies) / len(accuracies):.5f}, '
                            f'learning rate: {self.scheduler.get_last_lr()[0]:.8f}, {margin_str} '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', sum(loss_sum) / len(loss_sum), self.train_step)
                if self.train_method is not None:
                    writer.add_scalar('Train/il_loss', sum(il_losses) / len(il_losses), self.train_step)
                writer.add_scalar('Train/Accuracy', (sum(accuracies) / len(accuracies)), self.train_step)
                # 记录学习率
                writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], self.train_step)
                if self.margin_scheduler:
                    writer.add_scalar('Train/margin', self.margin_scheduler.get_margin(), self.train_step)
                self.train_step += 1
                train_times, accuracies, loss_sum = [], [], []
            # 固定步数也要保存一次模型
            if batch_id % 10000 == 0 and batch_id != 0 and local_rank == 0:
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id)
            start = time.time()
            self.scheduler.step()
            if self.margin_scheduler:
                self.margin_scheduler.step()

    def train(self,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None,
              do_eval=True,
              train_method=None,
              il_ratio=1.0):
        """
        训练模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        :param do_eval: 训练时是否评估模型
        :param train_method: 增量学习方法，为None则不使用增量学习
        :param il_ratio: 增量学习损失的权重
        """
        assert train_method in ['ewc', None]
        self.train_method = train_method
        if self.train_method is not None:
            logger.info(f'增量学习，增量学习方法: {train_method}')
        # 获取有多少张显卡训练
        nranks = torch.cuda.device_count()
        local_rank = 0
        writer = None
        if local_rank == 0:
            # 日志记录器
            writer = LogWriter(logdir='log')

        if nranks > 1 and self.use_gpu:
            # 初始化NCCL环境
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ["LOCAL_RANK"])
        # 获取数据
        self.__setup_dataloader(is_train=True)
        # 获取模型
        self.__setup_model(input_size=self.audio_featurizer.feature_dim, is_train=True)
        # 加载预训练模型
        self.__load_pretrained(pretrained_model=pretrained_model)
        # 获取原始的模型参数，用于计算增量学习损失
        if self.train_method is not None:
            for param_name, param in self.model.named_parameters():
                self.original_weight[param_name] = param.data.clone()
        # 加载恢复模型
        last_epoch, best_eer = self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)
        if last_epoch > 0:
            self.optimizer.step()
            [self.scheduler.step() for _ in range(last_epoch)]

        # 支持多卡训练
        if nranks > 1 and self.use_gpu:
            self.model.to(local_rank)
            self.audio_featurizer.to(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        logger.info('训练数据：{}'.format(len(self.train_dataset)))

        eer, min_dcf, threshold = None, None, None
        test_step, self.train_step = 0, 0
        last_epoch += 1
        if local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], last_epoch)
        # 开始训练
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            epoch_id += 1
            start_epoch = time.time()
            # 训练一个epoch
            self.__train_epoch(epoch_id=epoch_id, save_model_path=save_model_path, local_rank=local_rank,
                               writer=writer, nranks=nranks, il_ratio=il_ratio)
            # 多卡训练只使用一个进程执行评估和保存模型
            if local_rank == 0 and do_eval:
                logger.info('=' * 70)
                eer, min_dcf, threshold = self.evaluate()
                logger.info('Test epoch: {}, time/epoch: {}, threshold: {:.2f}, EER: {:.5f}, '
                            'MinDCF: {:.5f}'.format(epoch_id, str(timedelta(
                    seconds=(time.time() - start_epoch))), threshold, eer, min_dcf))
                logger.info('=' * 70)
                writer.add_scalar('Test/threshold', threshold, test_step)
                writer.add_scalar('Test/min_dcf', min_dcf, test_step)
                writer.add_scalar('Test/eer', eer, test_step)
                test_step += 1
                self.model.train()
                # # 保存最优模型
                if eer <= best_eer:
                    best_eer = eer
                    self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, eer=eer, min_dcf=min_dcf,
                                           threshold=threshold, best_model=True)
            if local_rank == 0:
                # 保存模型
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, eer=eer, min_dcf=min_dcf,
                                       threshold=threshold)

    def evaluate(self, resume_model=None, save_image_path=None):
        """
        评估模型
        :param resume_model: 所使用的模型
        :param save_image_path: 保存图片的路径
        :return: 评估结果
        """
        if self.enroll_loader is None or self.trials_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pth')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            if torch.cuda.is_available() and self.use_gpu:
                model_state_dict = torch.load(resume_model)
            else:
                model_state_dict = torch.load(resume_model, map_location='cpu')
            # 删除最后的分类层参数
            if self.model_output_name in model_state_dict.keys():
                del model_state_dict[self.model_output_name]
            self.model.load_state_dict(model_state_dict, strict=True)
            logger.info(f'成功加载模型：{resume_model}')
        self.model.eval()
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            eval_model = self.model.module if len(self.model.module) == 1 else self.model.module[0]
        else:
            eval_model = self.model if len(self.model) == 1 else self.model[0]

        # 获取注册的声纹特征和标签
        enroll_features, enroll_labels = None, None
        with torch.no_grad():
            for batch_id, (audio, label, input_lens_ratio) in enumerate(
                    tqdm(self.enroll_loader, desc="注册音频声纹特征")):
                audio = audio.to(self.device)
                input_lens_ratio = input_lens_ratio.to(self.device)
                label = label.to(self.device).long()
                audio_features, _ = self.audio_featurizer(audio, input_lens_ratio)
                feature = eval_model(audio_features).data.cpu().numpy()
                label = label.data.cpu().numpy()
                # 存放特征
                enroll_features = np.concatenate((enroll_features, feature)) if enroll_features is not None else feature
                enroll_labels = np.concatenate((enroll_labels, label)) if enroll_labels is not None else label
        # 获取检验的声纹特征和标签
        trials_features, trials_labels = None, None
        with torch.no_grad():
            for batch_id, (audio, label, input_lens_ratio) in enumerate(
                    tqdm(self.trials_loader, desc="验证音频声纹特征")):
                audio = audio.to(self.device)
                input_lens_ratio = input_lens_ratio.to(self.device)
                label = label.to(self.device).long()
                audio_features, _ = self.audio_featurizer(audio, input_lens_ratio)
                feature = eval_model(audio_features).data.cpu().numpy()
                label = label.data.cpu().numpy()
                # 存放特征
                trials_features = np.concatenate((trials_features, feature)) if trials_features is not None else feature
                trials_labels = np.concatenate((trials_labels, label)) if trials_labels is not None else label
        self.model.train()
        enroll_labels = enroll_labels.astype(np.int32)
        trials_labels = trials_labels.astype(np.int32)
        print('开始对比音频特征...')
        all_score, all_labels = [], []
        for i in tqdm(range(len(trials_features)), desc='特征对比'):
            trials_feature = np.expand_dims(trials_features[i], 0).repeat(len(enroll_features), axis=0)
            score = cosine_similarity(trials_feature, enroll_features).tolist()[0]
            trials_label = np.expand_dims(trials_labels[i], 0).repeat(len(enroll_features), axis=0)
            y_true = np.array(enroll_labels == trials_label).astype(np.int32).tolist()
            all_score.extend(score)
            all_labels.extend(y_true)
        # 计算EER
        all_score = np.array(all_score)
        all_labels = np.array(all_labels)
        fnr, fpr, thresholds = compute_fnr_fpr(all_score, all_labels)
        eer, threshold = compute_eer(fnr, fpr, all_score)
        min_dcf = compute_dcf(fnr, fpr)

        if save_image_path:
            import matplotlib.pyplot as plt
            index = np.where(np.array(thresholds) == threshold)[0][0]
            plt.plot(thresholds, fnr, color='blue', linestyle='-', label='fnr')
            plt.plot(thresholds, fpr, color='red', linestyle='-', label='fpr')
            plt.plot(threshold, fpr[index], 'ro-')
            plt.text(threshold, fpr[index], (round(threshold, 3), round(fpr[index], 5)), color='red')
            plt.xlabel('threshold')
            plt.title('fnr and fpr')
            plt.grid(True)  # 显示网格线
            # 保存图像
            os.makedirs(save_image_path, exist_ok=True)
            plt.savefig(os.path.join(save_image_path, 'result.png'))
            logger.info(f"结果图以保存在：{os.path.join(save_image_path, 'result.png')}")
        return eer, min_dcf, threshold

    def export(self, save_model_path='models/', resume_model='models/CAMPPlus_Fbank/best_model/'):
        """
        导出预测模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :return:
        """
        # 获取模型
        self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        # 加载预训练模型
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pth')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        model_state_dict = torch.load(resume_model)
        self.model.load_state_dict(model_state_dict)
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
        self.model.eval()
        # 获取静态模型
        infer_model = torch.jit.script(self.model)
        infer_model_path = os.path.join(save_model_path,
                                        f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                        'inference.pt')
        os.makedirs(os.path.dirname(infer_model_path), exist_ok=True)
        torch.jit.save(infer_model, infer_model_path)
        logger.info("预测模型已保存：{}".format(infer_model_path))
