from model import MusicTransformer
import custom
from custom.config import config

import torch

parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, [args.model_dir+'/save.yml']+args.configs, initialize=True)

# # check cuda
# if torch.cuda.is_available():
#     config.device = torch.device('cuda')
# else:
config.device = torch.device('cpu')

mt = MusicTransformer(
    embedding_dim=config.embedding_dim,
    vocab_size=config.vocab_size,
    num_layer=config.num_layers,
    max_seq=config.max_seq,
    dropout=0,
    debug=False)
mt.load_state_dict(torch.load(args.model_dir+'/final.pth'))
mt.test()

mt_script = torch.jit.trace(mt, (torch.rand(1,1), torch.tensor(100)))
print(mt_script.code)
