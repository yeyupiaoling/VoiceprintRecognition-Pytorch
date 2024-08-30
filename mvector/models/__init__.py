import importlib

from loguru import logger
from .campplus import CAMPPlus
from .ecapa_tdnn import EcapaTdnn
from .eres2net import ERes2Net, ERes2NetV2
from .fc import SpeakerIdentification
from .res2net import Res2Net
from .resnet_se import ResNetSE
from .tdnn import TDNN

__all__ = ['build_model']


def build_model(input_size, configs):
    use_model = configs.model_conf.get('model', 'CAMPPlus')
    model_args = configs.model_conf.get('model_args', {})
    mod = importlib.import_module(__name__)
    model = getattr(mod, use_model)(input_size=input_size, **model_args)
    logger.info(f'成功创建模型：{use_model}，参数为：{model_args}')
    return model
