from typing import Optional, Any

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss, _Loss
# from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule


class TransformerLoss(CrossEntropyLoss):
    def __init__(self, ignore_index=-100, reduction='mean') -> None:
        self.reduction = reduction
        self.ignore_index = ignore_index
        super().__init__(reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(torch.long)
        mask = (target != self.ignore_index).to(input.device, dtype=torch.long)
        not_masked_length = mask.to(torch.int).sum()
        input = input.permute(0, -1, -2)
        _loss = super().forward(input, target)
        _loss *= mask.to(_loss.dtype)
        return _loss.sum() / not_masked_length

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.forward(input, target)


class SmoothCrossEntropyLoss(_Loss):
    """
    https://arxiv.org/abs/1512.00567
    """
    __constants__ = ['label_smoothing', 'vocab_size', 'ignore_index', 'reduction']

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, reduction='mean', is_logits=True):
        assert 0.0 <= label_smoothing <= 1.0
        super().__init__(reduction=reduction)

        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.input_is_logits = is_logits

    def forward(self, input, target):
        """
        Args:
            input: [B * T, V]
            target: [B * T]
        Returns:
            cross entropy: [1]
        """
        mask = (target == self.ignore_index).unsqueeze(-1)
        q = F.one_hot(target.long(), self.vocab_size).type(torch.float32)
        u = 1.0 / self.vocab_size
        q_prime = (1.0 - self.label_smoothing) * q + self.label_smoothing * u
        q_prime = q_prime.masked_fill(mask, 0)

        ce = self.cross_entropy_with_logits(q_prime, input)
        if self.reduction == 'mean':
            lengths = torch.sum(target != self.ignore_index)
            return ce.sum() / lengths
        elif self.reduction == 'sum':
            return ce.sum()
        else:
            raise NotImplementedError

    def cross_entropy_with_logits(self, p, q):
        return -torch.sum(p * (q - q.logsumexp(dim=-1, keepdim=True)), dim=-1)


class CustomSchedule:
    def __init__(self, d_model, warmup_steps=4000, optimizer=None):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps

        self._step = 0
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.d_model ** (-0.5) * min(arg1, arg2)

