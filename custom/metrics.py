import torch
from typing import List


class _Metric(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        pass


class CategoricalAccuracy(_Metric):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        pass


class Accuracy(_Metric):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        pass


class MetricsSet(_Metric):
    def __init__(self, metrics: List[_Metric]):
        super().__init__()
        self.metrics = metrics

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return [metric(input, target) for metric in self.metrics]