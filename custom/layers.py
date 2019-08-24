import math as m
import numpy as np
import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def sinusoid(max_seq, embedding_dim):
    return np.array([[
        [
            m.sin(
                pos * m.exp(-m.log(10000) * i / embedding_dim) * m.exp(
                    m.log(10000) / embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
            )
            for i in range(embedding_dim)
        ]
        for pos in range(max_seq)
    ]])


class DynamicPositionEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, max_seq=2048):
        super().__init__()
        embed_sinusoid_list = np.array([[
            [
                m.sin(
                    pos * m.exp(-m.log(10000) * i/embedding_dim) *
                    m.exp(m.log(10000)/embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = embed_sinusoid_list

    def forward(self, x):
        x = x + Variable(self.positional_embedding[:, :x.size(1), :], requires_grad=False)
        return x


class RelativeGlobalAttention(torch.nn.Module):
    """
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    """
    def __init__(self, h=4, d=256, add_emb=False, max_seq=2048, **kwargs):
        super().__init__()
        self.len_k = None
        self.max_seq = max_seq
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = torch.nn.Linear(self.d, self.d)
        self.Wk = torch.nn.Linear(self.d, self.d)
        self.Wv = torch.nn.Linear(self.d, self.d)
        self.fc = torch.nn.Linear(d, d)
        self.additional = add_emb
        self.E = self.add_weight('emb', shape=[self.max_seq, int(self.dh)])
        if self.additional:
            self.Radd = None

    def forward(self, inputs, mask=None, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs[0]
        q = self.Wq(q)
        q = torch.Tensor.reshape(q, (q.size(0), q.size(1), self.h, -1))
        q = torch.Tensor.transpose(q, (0, 2, 1, 3))  # batch, h, seq, dh

        k = inputs[1]
        k = self.Wk(k)
        k = torch.Tensor.reshape(k, (k.size(0), k.size(1), self.h, -1))
        k = torch.Tensor.transpose(k, (0, 2, 1, 3))

        v = inputs[2]
        v = self.Wv(v)
        v = torch.Tensor.reshape(v, (v.size(0), v.size(1), self.h, -1))
        v = torch.Tensor.transpose(v, (0, 2, 1, 3))

        self.len_k = k.shape[2]
        self.len_q = q.shape[2]

        E = self._get_left_embedding(self.len_q, self.len_k)
        QE = torch.Tensor.einsum('bhld,md->bhlm', q, E)
        QE = self._qe_masking(QE)
        # print(QE.shape)
        Srel = self._skewing(QE)

        Kt = torch.Tensor.transpose(k,[0, 1, 3, 2])
        QKt = torch.Tensor.matmul(q, Kt)
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits += (torch.Tensor.cast(mask, torch.Tensor.float) * -1e9)

        attention_weights = F.softmax(logits, -1)
        # tf.print('logit result: \n', logits, output_stream=sys.stdout)
        attention = torch.Tensor.matmul(attention_weights, v)
        # tf.print('attention result: \n', attention, output_stream=sys.stdout)

        out = torch.Tensor.transpose(attention, (0, 2, 1, 3))
        out = torch.Tensor.reshape(out, (out.size(0), -1, self.d))

        out = self.fc(out)
        return out, attention_weights

    def _get_left_embedding(self, len_q, len_k):
        starting_point = max(0,self.max_seq-len_q)
        e = self.E[starting_point:,:]
        return e

    # @staticmethod
    # def _qe_masking(qe):
    #     mask = tf.sequence_mask(
    #         tf.range(qe.shape[-1] -1, qe.shape[-1] - qe.shape[-2] -1, -1), qe.shape[-1])
    #
    #     mask = tf.logical_not(mask)
    #     mask = tf.cast(mask, tf.float32)
    #
    #     return mask * qe

    def _skewing(self, tensor: torch.Tensor):
        padded = torch.Tensor.pad(tensor, [[0, 0], [0,0], [0, 0], [1, 0]])
        reshaped = torch.Tensor.reshape(padded, shape=[-1, padded.size(1), padded.size(-1), padded.size(-2)])
        Srel = reshaped[:, :, 1:, :]
        # print('Sre: {}'.format(Srel))

        if self.len_k > self.len_q:
            Srel = torch.Tensor.pad(Srel, [[0,0], [0,0], [0,0], [0, self.len_k-self.len_q]])
        elif self.len_k < self.len_q:
            Srel = Srel[:,:,:,:self.len_k]

        return Srel


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(h=h, d=d_model, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = torch.nn.Linear(self.d_model, self.d_model//2)
        self.FFN_suf = torch.nn.Linear(self.d_model//2, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask=None, **kwargs):
        attn_out, w = self.rga([x,x,x], mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)

        ffn_out = torch.nn.ReLU(self.FFN_pre(out1))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1+ffn_out)
        return out2, w


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.rga2 = RelativeGlobalAttention(d=d_model, h=h, max_seq=max_seq, add_emb=additional)
        self.rga = RelativeGlobalAttention(d=d_model, h=h, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = torch.nn.Linear(self.d_model, self.d_model // 2)
        self.FFN_suf = torch.nn.Linear(self.d_model // 2, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm3 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.dropout3 = torch.nn.Dropout(rate)

    def forward(self, x, encode_out, mask=None, lookup_mask=None, w_out=False, **kwargs):

        attn_out, aw1 = self.rga([x, x, x], mask=lookup_mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)

        if encode_out is None:
            attn_out2, aw2 = self.rga2([out1, out1, out1], mask=mask)
        else:
            attn_out2, aw2 = self.rga2([out1, encode_out, encode_out], mask=mask)
        attn_out2 = self.dropout2(attn_out2)
        attn_out2 = self.layernorm2(out1+attn_out2)

        ffn_out = torch.nn.ReLU(self.FFN_pre(attn_out2))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout3(ffn_out)
        out = self.layernorm3(attn_out2+ffn_out)

        if w_out:
            return out, aw1, aw2
        else:
            return out