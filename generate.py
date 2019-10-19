from custom.layers import *
from custom import criterion
from data import Data
from custom.config import config
import utils
from midi_processor.processor import decode_midi, encode_midi

import datetime
import argparse

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
args = parser.parse_args()

config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')


current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
gen_summary_writer = SummaryWriter(gen_log_dir)

# mt = MusicTransformer(
#             embedding_dim=256,
#             vocab_size=par.vocab_size,
#             num_layer=6,
#             max_seq=2048,
#             dropout=0.2,
#             debug=False, loader_path=load_path)
mt = torch.load(config.load_path)
mt.eval()

if config.condition_file is not None:
    inputs = np.array([encode_midi('dataset/midi/BENABD10.mid')[:500]])
else:
    inputs = np.array([[28]])
inputs = torch.from_numpy([inputs]).to(config.device)

result = mt.generate(inputs, beam=1, length=config.length, tf_board_writer=gen_summary_writer)

for i in result:
    print(i)

decode_midi(result, file_path=config.save_path)

gen_summary_writer.close()
