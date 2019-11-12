import sys
import os
sys.path.append(os.path.abspath('.'))

from model import MusicTransformer
import custom
from custom.metrics import *
from custom.criterion import SmoothCrossEntropyLoss, CustomSchedule, TransformerLoss
from custom.config import config
from custom.parallel import DataParallelModel, DataParallelCriterion
from data import Data

import utils
import datetime
import time
import pprint

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
    device_ids = list(range(torch.cuda.device_count()))
else:
    config.device = torch.device('cpu')
    device_ids = None

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
opt = optim.Adam(mt.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = CustomSchedule(config.embedding_dim, optimizer=opt)

# multi-GPU set
if torch.cuda.device_count() > 1:
    single_mt = mt
    mt = DataParallelModel(mt)
    mt.cuda()
else:
    single_mt = mt

# init metric set
metrics_dict = {
    'accuracy': CategoricalAccuracy(),
    'loss': TransformerLoss(),
    'bucket':  LogitsBucketting(config.vocab_size)
}
metric_set = ParallelMetricSet(metrics_dict)
eval_metric_set = MetricsSet(metrics_dict)


print(mt)
print('| Summary - Available Device Count Info : {}'.format(torch.cuda.device_count()))

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/'+config.experiment+'/'+current_time+'/train'
eval_log_dir = 'logs/'+config.experiment+'/'+current_time+'/eval'

train_summary_writer = SummaryWriter(train_log_dir)
eval_summary_writer = SummaryWriter(eval_log_dir)

# Train Start
print(">> Train start...")
idx = 0
for e in range(config.epochs):
    print(">>> [Epoch was updated]")
    for b in range(len(dataset.files) // config.batch_size):
        scheduler.optimizer.zero_grad()
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(config.batch_size, config.max_seq)
            batch_x = torch.from_numpy(batch_x).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
            batch_y = torch.from_numpy(batch_y).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
        except IndexError:
            continue

        start_time = time.time()
        mt.train()
        sample = mt(batch_x)
        # pprint.pprint(sample)
        metrics = metric_set(sample, batch_y)
        loss = metrics['loss']
        loss.backward()
        scheduler.step()
        end_time = time.time()

        if config.debug:
            print("[Loss]: {}".format(loss))

        train_summary_writer.add_scalar('loss', metrics['loss'], global_step=idx)
        train_summary_writer.add_scalar('accuracy', metrics['accuracy'], global_step=idx)
        train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=idx)
        train_summary_writer.add_scalar('iter_p_sec', end_time-start_time, global_step=idx)

        # result_metrics = metric_set(sample, batch_y), run single gpu on eval time.
        if b % 100 == 0:
            single_mt.eval()
            eval_x, eval_y = dataset.slide_seq2seq_batch(2, config.max_seq, 'eval')
            eval_x = torch.from_numpy(eval_x).contiguous().to(config.device, dtype=torch.int)
            eval_y = torch.from_numpy(eval_y).contiguous().to(config.device, dtype=torch.int)

            eval_preiction, weights = single_mt.forward(eval_x)

            eval_metrics = eval_metric_set(eval_preiction, eval_y)
            torch.save(single_mt.state_dict(), args.model_dir+'/train-{}.pth'.format(e))
            if b == 0:
                train_summary_writer.add_histogram("target_analysis", batch_y, global_step=e)
                train_summary_writer.add_histogram("source_analysis", batch_x, global_step=e)
                for i, weight in enumerate(weights):
                    attn_log_name = "attn/layer-{}".format(i)
                    utils.attention_image_summary(
                        attn_log_name, weight, step=idx, writer=eval_summary_writer)

            eval_summary_writer.add_scalar('loss', eval_metrics['loss'], global_step=idx)
            eval_summary_writer.add_scalar('accuracy', eval_metrics['accuracy'], global_step=idx)
            eval_summary_writer.add_histogram("logits_bucket", eval_metrics['bucket'], global_step=idx)

            print('\n====================================================')
            print('Epoch/Batch: {}/{}'.format(e, b))
            print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(metrics['loss'], metrics['accuracy']))
            print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_metrics['loss'], eval_metrics['accuracy']))
        idx += 1

torch.save(single_mt.state_dict(), args.model_dir+'/final.pth'.format(idx))
eval_summary_writer.close()
train_summary_writer.close()