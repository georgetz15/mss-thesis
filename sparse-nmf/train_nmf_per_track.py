import argparse
import json

from datasets import MUSDBDataset
import torch
from nmf import NMF, concat_dictionaries
from pathlib import Path
import tqdm
import scipy.signal as ss
import numpy as np


def stft(x, sr=44100, n_fft=2048, n_hop=512):
    return ss.stft(x, sr,
                   nperseg=n_fft,
                   noverlap=n_fft - n_hop,
                   nfft=n_fft)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-track NMF trainer')

    parser.add_argument('--out-dir', type=str, help='Directory where the individual models will be saved.')
    parser.add_argument('--musdb-root', type=str, help='root path of dataset', required=True)
    parser.add_argument('--is-wav', action='store_true', default=True,
                        help='Uses musdb wav representation.')
    parser.add_argument('--target', type=str, default='vocals',
                        help='target source (will be passed to the dataset)')
    parser.add_argument('--n-workers', type=int, default=2,
                        help='Number of workers for dataloader.')
    parser.add_argument('--sr', type=float, default=44100,
                        help='Sampling rate of the tracks.')

    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='Choose device for training')
    parser.add_argument('--dtype', type=str, default='f32', choices=['f32', 'f64'],
                        help='Choose dtype for training')

    parser.add_argument('--n-fft', type=int, default=2048)
    parser.add_argument('--n-hop', type=int, default=512)
    parser.add_argument('--tol', type=float, default=1e-5, help='Cost change tolerance for stopping criterion.')
    parser.add_argument('--n-components', type=int, default=20, help='# of components per track.')
    parser.add_argument('--n-stacked-frames', type=int, default=6, help='# of stacked frames.')
    parser.add_argument('--sparsity-weight', type=float, default=0,
                        help='regularization weight, defaults to 0')
    parser.add_argument('--b-div', type=float, default=1, help='Beta divergence to be used.')
    parser.add_argument('--specgram-pow', type=float, default=1,
                        help='Power of abs(stft(x))')
    parser.add_argument('--silence-thres-db', type=float, default=-60,
                        help='Silence threshold to filter fft frames.')

    args, _ = parser.parse_known_args()

    out_dir = Path(args.out_dir).expanduser() / args.target
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dtype == 'f32':
        dtype = torch.float32
    elif args.dtype == 'f64':
        dtype = torch.float64
    else:
        ValueError('args.dtype should be f64 or f32.')

    # musdb_root = Path(args.musdb_root).expanduser()
    musdb_kwargs = {
        'subsets': 'train',
        'split': 'train'
    }
    dataset = MUSDBDataset(args.musdb_root, args.is_wav, args.target, args.n_fft, args.n_hop, args.sr, args.n_stacked_frames,
                                      dtype, args.device,
                                      args.specgram_pow, args.silence_thres_db, transform=True, **musdb_kwargs)

    F = (args.n_fft // 2 + 1) * args.n_stacked_frames
    K = args.n_components
    b = args.b_div

    def generate_model():
        return NMF(F, K, b=args.b_div, m=args.sparsity_weight,
                                 robust_normalization=True, tol=args.tol, dtype=dtype,
                                 device=args.device, keep_history=True)

    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': args.n_workers,
              'pin_memory': True}
    dataset_gen = torch.utils.data.DataLoader(dataset, **params)

    pbar = tqdm.tqdm(dataset_gen)
    pbar.set_description(f"Training {args.target} track")
    ix = 0
    for V in pbar:
        model = generate_model()
        V = torch.tensor(V[0], device=args.device, dtype=dtype)
        model.train(V)
        torch.save(model, out_dir / f'nmf_{ix}.pt')
        ix += 1

    concat_dictionaries(out_dir)

    # save params
    params = {
        'args': vars(args),
    }

    with open(out_dir / f'{args.target}.json', 'w') as outfile:
        outfile.write(json.dumps(params, indent=4, sort_keys=True))
