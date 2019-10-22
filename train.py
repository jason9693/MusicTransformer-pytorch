from model import MusicTransformer
import custom
from custom.metrics import *
from custom.layers import *
from custom.criterion import SmoothCrossEntropyLoss
from custom.config import config
from data import Data

import utils
import argparse
import datetime

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter


# set config
parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')


# load data
dataset = Data(config.pickle_dir)
print(dataset)


# load model
learning_rate = config.l_r

# define model
mt = MusicTransformer(
            embedding_dim=config.embedding_dim,
            vocab_size=config.vocab_size,
            num_layer=config.num_layers,
            max_seq=config.max_seq,
            dropout=config.dropout,
            debug=config.debug, loader_path=config.load_path
)
mt.to(config.device)
opt = optim.Adam(mt.parameters(), lr=config.l_r)
metric_set = MetricsSet({
    'accuracy': CategoricalAccuracy(),
    'loss': SmoothCrossEntropyLoss(config.label_smooth, config.vocab_size, config.pad_token)
})

# multi-GPU set
if torch.cuda.device_count() > 1:
    single_mt = mt
    mt = torch.nn.DataParallel(mt)
else:
    single_mt = mt

print(mt)

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/mt_decoder/'+current_time+'/train'
eval_log_dir = 'logs/mt_decoder/'+current_time+'/eval'

train_summary_writer = SummaryWriter(train_log_dir)
eval_summary_writer = SummaryWriter(eval_log_dir)

# Train Start
print(">> Train start...")
idx = 0
for e in range(config.epochs):
    print(">>> [Epoch was updated]")
    for b in range(len(dataset.files) // config.batch_size):
        opt.zero_grad()
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(config.batch_size, config.max_seq)
            batch_x = torch.from_numpy(batch_x).contiguous().to(config.device, non_blocking=True)
            batch_y = torch.from_numpy(batch_y).contiguous().to(config.device, non_blocking=True)
        except IndexError:
            continue

        mt.train()
        sample, _ = mt.forward(batch_x)
        metrics = metric_set(sample, batch_y)
        loss = metrics['loss']
        loss.backward()
        opt.step()
        if config.debug:
            print("[Loss]: {}".format(loss))

        # result_metrics = metric_set(sample, batch_y)
        if b % 100 == 0:
            eval_x, eval_y = dataset.slide_seq2seq_batch(config.batch_size, config.max_seq, 'eval')
            eval_x = torch.from_numpy(eval_x).contiguous().to(config.device, non_blocking=True)
            eval_y = torch.from_numpy(eval_y).contiguous().to(config.device, non_blocking=True)

            eval_preiction, weights = mt.forward(eval_x)
            eval_metrics = metric_set(eval_preiction, eval_y)
            torch.save(single_mt.state_dict(), args.model_dir+'/train-{}.pth'.format(idx))
            if b == 0:
                train_summary_writer.add_histogram("target_analysis", batch_y, global_step=e)
                train_summary_writer.add_histogram("source_analysis", batch_x, global_step=e)
            train_summary_writer.add_scalar('loss', metrics['loss'], global_step=idx)
            train_summary_writer.add_scalar('accuracy', metrics['accuracy'], global_step=idx)

            eval_summary_writer.add_scalar('loss', eval_metrics['loss'], global_step=idx)
            eval_summary_writer.add_scalar('accuracy', eval_metrics['accuracy'], global_step=idx)
            for i, weight in enumerate(weights):
                    attn_log_name = "attn/layer-{}".format(i)
                    utils.attention_image_summary(attn_log_name, weights, step=idx, writer=eval_summary_writer)
            idx += 1
            print('\n====================================================')
            print('Epoch/Batch: {}/{}'.format(e, b))
            print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(metrics['loss'], metrics['accuracy']))
            print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_metrics['loss'], eval_metrics['acccuracy']))

torch.save(single_mt.state_dict(), args.model_dir+'/final.pth'.format(idx))
eval_summary_writer.close()
train_summary_writer.close()


