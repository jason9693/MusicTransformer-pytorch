from model import MusicTransformer
from custom.metrics import *
from custom.layers import *
from custom.criterion import TransformerLoss
import params as par
from data import Data
import utils
import argparse
import datetime
import torch.optim as optim
from tensorboardX import SummaryWriter
import sys


parser = argparse.ArgumentParser()

parser.add_argument('--l_r', default=None, help='학습률', type=float)
parser.add_argument('--batch_size', default=2, help='batch size', type=int)
parser.add_argument('--pickle_dir', default='music', help='데이터셋 경로')
parser.add_argument('--max_seq', default=2048, help='최대 길이', type=int)
parser.add_argument('--epochs', default=100, help='에폭 수', type=int)
parser.add_argument('--load_path', default=None, help='모델 로드 경로', type=str)
parser.add_argument('--save_path', default="result/dec0722", help='모델 저장 경로')
parser.add_argument('--is_reuse', default=False)
parser.add_argument('--multi_gpu', default=True)
parser.add_argument('--num_layers', default=6, type=int)

args = parser.parse_args()


# set arguments
l_r = args.l_r
batch_size = args.batch_size
pickle_dir = args.pickle_dir
max_seq = args.max_seq
epochs = args.epochs
is_reuse = args.is_reuse
load_path = args.load_path
save_path = args.save_path
multi_gpu = args.multi_gpu
num_layer = args.num_layers


# load data
dataset = Data('dataset/processed')
print(dataset)


# load model
learning_rate = l_r

# define model
mt = MusicTransformer(
            embedding_dim=256,
            vocab_size=par.vocab_size,
            num_layer=num_layer,
            max_seq=max_seq,
            dropout=0.2,
            debug=False, loader_path=load_path
)
criterion = TransformerLoss
opt = optim.Adam(mt.parameters(), lr=l_r)
metric_set = MetricsSet([Accuracy, ])

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/mt_decoder/'+current_time+'/train'
eval_log_dir = 'logs/mt_decoder/'+current_time+'/eval'

train_summary_writer = SummaryWriter(train_log_dir)
eval_summary_writer = SummaryWriter(eval_log_dir)


# Train Start
idx = 0
opt.zero_grad()
for e in range(epochs):
    for b in range(len(dataset.files) // batch_size):
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(batch_size, max_seq)
            batch_x = torch.from_numpy(batch_x)
            batch_y - torch.from_numpy(batch_y)
        except:
            continue

        sample = mt.train_forward(batch_x)
        loss = criterion(sample, batch_y)
        loss.backward()
        opt.step()
        
        result_metrics = metric_set(sample, batch_y)
        if b % 100 == 0:
            eval_x, eval_y = dataset.slide_seq2seq_batch(batch_size, max_seq, 'eval')
            eval_result_metrics, weights = mt.evaluate(eval_x, eval_y)
            mt.save(save_path)
            if b == 0:
                train_summary_writer.add_histogram("target_analysis", batch_y, global_step=e)
                train_summary_writer.add_histogram("source_analysis", batch_x, global_step=e)
            train_summary_writer.add_scalar('loss', result_metrics[0], global_step=idx)
            train_summary_writer.add_scalar('accuracy', result_metrics[1], global_step=idx)

            eval_summary_writer.add_scalar('loss', eval_result_metrics[0], global_step=idx)
            eval_summary_writer.add_scalar('accuracy', eval_result_metrics[1], global_step=idx)
            for i, weight in enumerate(weights):
                    attn_log_name = "attn/layer-{}".format(i)
                    utils.attention_image_summary(attn_log_name, step=idx)
            idx += 1
            print('\n====================================================')
            print('Epoch/Batch: {}/{}'.format(e, b))
            print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(result_metrics[0], result_metrics[1]))
            print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_result_metrics[0], eval_result_metrics[1]))


