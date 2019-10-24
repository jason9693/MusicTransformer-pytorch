import pickle
import os
import sys
from progress.bar import Bar
import utils
from midi_processor.processor import encode_midi


def preprocess_midi(path):
    return encode_midi(path)


def preprocess_midi_files_under(midi_root, save_dir):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'

    for path in Bar('Processing').iter(midi_paths):
        print(' ', end='[{}]'.format(path), flush=True)

        try:
            data = preprocess_midi(path)
        except KeyboardInterrupt:
            print(' Abort')
            return
        except EOFError:
            print('EOF Error')
            return

        with open('{}/{}.pickle'.format(save_dir, path.split('/')[-1]), 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    preprocess_midi_files_under(
            midi_root=sys.argv[1],
            save_dir=sys.argv[2])
