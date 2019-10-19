from custom.layers import *
from custom.criterion import *
from custom.layers import Encoder

import sys
import torch
import torch.distributions as dist
import random
import utils

import torch
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar


class MusicTransformer(torch.nn.Module):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, debug=False, loader_path=None, dist=False, writer=None):
        super().__init__()

        if loader_path is not None:
            self.load_config_file(loader_path)
        else:
            self._debug = debug
            self.max_seq = max_seq
            self.num_layer = num_layer
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size
            self.dist = dist

        self.writer = writer
        self.Decoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.fc = torch.nn.Linear(self.embedding_dim, self.vocab_size)

        self._set_metrics()

    def forward(self, x, lookup_mask=None):
        decoder, w = self.Decoder(x, mask=lookup_mask)
        fc = self.fc(decoder)
        fc = fc.softmax(-1)
        return fc, w

    def generate(self, prior: torch.Tensor, length=2048, tf_board_writer: SummaryWriter = None):
        decode_array = prior
        for i in Bar('generating').iter(range(min(self.max_seq, length))):
            if decode_array.shape[1] >= self.max_seq:
                break
            if i % 100 == 0:
                print('generating... {}% completed'.format((i / min(self.max_seq, length)) * 100))
            _, _, look_ahead_mask = \
                utils.get_masked_with_pad_tensor(decode_array.shape[1], decode_array, decode_array)

            result, _ = self.forward(decode_array, lookup_mask=look_ahead_mask)
            if tf_board_writer:
                tf_board_writer.add_image("logits", result, global_step=i)

            u = random.uniform(0, 1)
            if u > 1:
                result = result[:, -1].argmax(-1).to(torch.int32)
                decode_array = torch.cat([decode_array, result.unsqueeze(-1)], -1)
            else:
                pdf = dist.OneHotCategorical(probs=result[:, -1])
                result = pdf.sample(1)
                result = torch.transpose(result, 1, 0).to(torch.int32)
                decode_array = torch.cat((decode_array, result), dim=-1)
            del look_ahead_mask
        decode_array = decode_array[0]
        return decode_array

    def teacher_forcing_forward(self, x, attn=False):
        x, _ = self.__prepare_train_data(x, x)
        _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x)

        predictions, w = self.forward(
            x, lookup_mask=look_ahead_mask,
        )

        if self._debug:
            print('train step finished')
        if attn:
            return predictions, w
        else:
            return predictions
