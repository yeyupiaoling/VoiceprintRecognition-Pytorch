import math
from typing import List


class WarmupCosineSchedulerLR:
    def __init__(
            self,
            optimizer,
            min_lr,
            max_lr,
            warmup_epoch,
            fix_epoch,
            step_per_epoch
    ):
        self.optimizer = optimizer
        assert min_lr <= max_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_step = warmup_epoch * step_per_epoch
        self.fix_step = fix_epoch * step_per_epoch
        self.current_step = 0.0

    def set_lr(self, ):
        new_lr = self.clr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def step(self, step=None):
        if step is not None:
            self.current_step = step
        new_lr = self.set_lr()
        self.current_step += 1
        return new_lr

    def clr(self, step):
        if step < self.warmup_step:
            return self.min_lr + (self.max_lr - self.min_lr) * \
                (step / self.warmup_step)
        elif self.warmup_step <= step < self.fix_step:
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                (1 + math.cos(math.pi * (step - self.warmup_step) /
                              (self.fix_step - self.warmup_step)))
        else:
            return self.min_lr

    def get_last_lr(self) -> List[float]:
        return [self.clr(self.current_step)]


class MarginScheduler:
    def __init__(
            self,
            criterion,
            increase_start_epoch,
            fix_epoch,
            step_per_epoch,
            initial_margin=0.0,
            final_margin=0.3,
            increase_type='exp',
    ):
        assert hasattr(criterion, 'update'), "Loss function not has 'update()' attributes."
        self.criterion = criterion
        self.increase_start_step = increase_start_epoch * step_per_epoch
        self.fix_step = fix_epoch * step_per_epoch
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.increase_type = increase_type
        self.margin = initial_margin

        self.current_step = 0
        self.increase_step = self.fix_step - self.increase_start_step

        self.init_margin()

    def init_margin(self):
        self.criterion.update(margin=self.initial_margin)

    def step(self, current_step=None):
        if current_step is not None:
            self.current_step = current_step

        self.margin = self.iter_margin()
        self.criterion.update(margin=self.margin)
        self.current_step += 1

    def iter_margin(self):
        if self.current_step < self.increase_start_step:
            return self.initial_margin

        if self.current_step >= self.fix_step:
            return self.final_margin

        a = 1.0
        b = 1e-3

        current_step = self.current_step - self.increase_start_step
        if self.increase_type == 'exp':
            # exponentially increase the margin
            ratio = 1.0 - math.exp(
                (current_step / self.increase_step) *
                math.log(b / (a + 1e-6))) * a
        else:
            # linearly increase the margin
            ratio = 1.0 * current_step / self.increase_step
        return self.initial_margin + (self.final_margin -
                                      self.initial_margin) * ratio

    def get_margin(self):
        return self.margin
