from typing import Optional, Any

import params as par
import sys

from torch.__init__ import Tensor
import torch
from torch.nn.modules.loss import CrossEntropyLoss
# from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule


class MTFitCallback(keras.callbacks.Callback):

    def __init__(self, save_path):
        super(MTFitCallback, self).__init__()
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.save_path)


class TransformerLoss(CrossEntropyLoss):
    def __init__(self, weight: Optional[Any] = ..., ignore_index: int = ..., reduction: str = ...) -> None:
        self.reduction = reduction
        super().__init__(weight, ignore_index, 'none')

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mask = target != par.pad_token
        not_masked_length = mask.to(torch.int).sum()
        input = input.permute(0, -1, -2)
        _loss = super().forward(input, target)
        _loss *= mask
        return _loss.sum() / not_masked_length

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return self.forward(input, target)


def transformer_dist_train_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.math.logical_not(tf.math.equal(y_true, par.pad_token))
    mask = tf.cast(mask, tf.float32)

    y_true_vector = tf.one_hot(y_true, par.vocab_size)

    _loss = tf.nn.softmax_cross_entropy_with_logits(y_true_vector, y_pred)
    # print(_loss.shape)
    #
    # _loss = tf.reduce_mean(_loss, -1)
    _loss *= mask

    return _loss


class CustomSchedule(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def get_config(self):
        super(CustomSchedule, self).get_config()

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == '__main__':
    import numpy as np
    loss = TransformerLoss()(np.array([[1],[0],[0]]), tf.constant([[0.5,0.5],[0.1,0.1],[0.1,0.1]]))
    print(loss)
