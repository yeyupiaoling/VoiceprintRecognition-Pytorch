import importlib

from .aamloss import AAMLoss
from .amloss import AMLoss
from .armloss import ARMLoss
from .celoss import CELoss
from .sphereface2 import SphereFace2
from .subcenterloss import SubCenterLoss
from .tripletangularmarginloss import TripletAngularMarginLoss
from mvector.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_loss(configs):
    use_loss = configs.loss_conf.get('loss', 'AAMLoss')
    loss_args = configs.loss_conf.get('loss_args', {})
    los = importlib.import_module(__name__)
    loss = getattr(los, use_loss)(**loss_args)
    logger.info(f'成功创建损失函数：{use_loss}，参数为：{loss_args}')
    return loss
