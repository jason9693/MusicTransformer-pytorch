import torch
import torch.nn.functional as F

from typing import Dict


class _Metric(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        raise NotImplementedError()


class Accuracy(_Metric):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        bool_acc = input.long() == target.long()
        return bool_acc.sum() / bool_acc.numel()


class CategoricalAccuracy(Accuracy):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        categorical_input = input.argmax(-1)
        return super().forward(categorical_input, target)


class MetricsSet(object):
    def __init__(self, metric_dict: Dict):
        super().__init__()
        self.metrics = metric_dict

    def __call__(self, input: torch.Tensor, target: torch.Tensor):
        return self.forward(input=input, target=target)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # return [metric(input, target) for metric in self.metrics]
        return {k: metric(input, target) for k, metric in self.metrics.items()}
