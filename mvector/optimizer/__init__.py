import importlib

from torch.optim import *
from .scheduler import WarmupCosineSchedulerLR
from torch.optim.lr_scheduler import *
from mvector.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_optimizer(params, configs):
    use_optimizer = configs.optimizer_conf.get('optimizer', 'Adam')
    optimizer_args = configs.optimizer_conf.get('optimizer_args', {})
    optim = importlib.import_module(__name__)
    optimizer = getattr(optim, use_optimizer)(params=params, **optimizer_args)
    logger.info(f'成功创建优化方法：{use_optimizer}，参数为：{optimizer_args}')
    return optimizer


def build_lr_scheduler(optimizer, step_epoch, configs):
    use_scheduler = configs.optimizer_conf.get('scheduler', 'WarmupCosineSchedulerLR')
    scheduler_args = configs.optimizer_conf.get('scheduler_args', {})
    if configs.optimizer_conf.scheduler == 'CosineAnnealingLR':
        scheduler_args.T_max = int(configs.train_conf.max_epoch * 1.2) * step_epoch
    elif configs.optimizer_conf.scheduler == 'WarmupCosineSchedulerLR':
        scheduler_args.fix_epoch = configs.train_conf.max_epoch
        scheduler_args.step_per_epoch = configs.train_conf.max_epoch
    optim = importlib.import_module(__name__)
    scheduler = getattr(optim, use_scheduler)(optimizer=optimizer, **scheduler_args)
    logger.info(f'成功创建学习率衰减：{use_scheduler}，参数为：{scheduler_args}')
    return scheduler
