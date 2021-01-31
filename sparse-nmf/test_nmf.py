import argparse
from pathlib import Path

import musdb
import museval
import torch
import tqdm
import scipy.signal as ss
import numpy as np

from nmf import *


def stft(x, sr=44100, n_fft=2048, n_hop=512):
    return ss.stft(x, sr,
                   nperseg=n_fft,
                   noverlap=n_fft - n_hop,
                   nfft=n_fft)


def istft(X, sr, n_fft, n_hop):
    _, x = ss.istft(
        X,
        sr,
        nperseg=n_fft,
        noverlap=n_fft - n_hop
    )
    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Use this script to evaluate a sparse nmf model on vox/acc separation'
                                                 ' using the MUSDB18 test dataset.')

    parser.add_argument('--out-dir', type=str, help='Directory where the individual models will be saved.')
    parser.add_argument('--musdb-root', type=str, help='root path of dataset', required=True)
    parser.add_argument('--is-wav', action='store_true', default=True,
                        help='Uses musdb wav representation.')
    parser.add_argument('--n-workers', type=int, default=2,
                        help='Number of workers for dataloader.')
    parser.add_argument('--sr', type=float, default=44100,
                        help='Sampling rate of the tracks.')

    parser.add_argument('--acc', type=str, help='Path to accompaniment model', required=True)
    parser.add_argument('--vox', type=str, help='Path to vocals model', required=True)

    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='Choose device for training')
    parser.add_argument('--dtype', type=str, default='f32', choices=['f32', 'f64'],
                        help='Choose dtype for training')

    parser.add_argument('--n-fft', type=int, default=2048)
    parser.add_argument('--n-hop', type=int, default=512)
    parser.add_argument('--tol', type=float, default=1e-5, help='Cost change tolerance for stopping criterion.')
    parser.add_argument('--n-stacked-frames', type=int, default=6, help='# of stacked frames.')
    parser.add_argument('--sparsity-weight', type=float, default=0,
                        help='regularization weight, defaults to 0')
    parser.add_argument('--b-div', type=float, default=1, help='Beta divergence to be used.')
    parser.add_argument('--specgram-pow', type=float, default=1,
                        help='Power of abs(stft(x))')

    args, _ = parser.parse_known_args()
    out_dir = Path(args.out_dir).expanduser()

    if args.dtype == 'f32':
        dtype = torch.float32
    elif args.dtype == 'f64':
        dtype = torch.float32
    else:
        ValueError('args.dtype should be f64 or f32.')

    musdb_kwargs = {
        'subsets': 'test'
    }

    mus = musdb.DB(args.musdb_root, is_wav=args.is_wav, subsets='test')

    model_vox = torch.load(Path(args.vox).expanduser())
    vox_ix = slice(0, model_vox.K)
    model_acc = torch.load(Path(args.acc).expanduser())
    acc_ix = slice(model_vox.K, model_vox.K + model_acc.K)
    model_vox.W = torch.cat([model_vox.W, model_acc.W], dim=1)
    model = model_vox
    model.b = args.b_div
    model.m = args.sparsity_weight
    model.tol = args.tol
    model.to(args.device)
    del model_acc
    # TODO: set dtype

    n_sources = 2
    pbar = tqdm(mus)
    pbar.set_description(f"Separating track")
    results = museval.EvalStore()  # evaluation results object. Will be used to save results in pandas df.
    for track in pbar:
        x = track.targets['linear_mixture'].audio.T
        track_len = x.shape[1]

        # Note: You may want to use this code block if you have <= 16 GB of RAM. Eval is memory hungry :)
        # if (track.duration / 60 > 5.7):
        #     print(f'Skipping {track.name}, with duration {track.duration / 60} min.')
        #     continue

        n_channels, n_samples = x.shape
        vox = track.targets['vocals'].audio.T
        acc = track.targets['accompaniment'].audio.T
        X = stft(x, args.sr, args.n_fft, args.n_hop)[2]

        # Preprocessing
        V = np.abs(X) ** args.specgram_pow
        # Stack frames
        stacked = []
        for i in range(n_channels):
            stacked.append(stack_frames(V[i], args.n_stacked_frames))
        V = torch.tensor(stacked, device=args.device, dtype=dtype)
        s = np.zeros((n_sources, n_channels, n_samples))
        for i in range(n_channels):
            H = model.infer(V[i] + 1e-12)
            Vhat = model.W @ H
            Vvox = model.W[:, vox_ix] @ H[vox_ix, :]
            Vacc = model.W[:, acc_ix] @ H[acc_ix, :]
            Gvox = (Vvox / Vhat).cpu().numpy()
            Gvox = unstack_frames(Gvox, args.n_stacked_frames)
            Gacc = (Vacc / Vhat).cpu().numpy()
            Gacc = unstack_frames(Gacc, args.n_stacked_frames)
            s[0, i] = istft(Gvox[:, :X.shape[-1]] * X[i], args.sr, args.n_fft, args.n_hop)[:track_len]
            s[1, i] = istft(Gacc[:, :X.shape[-1]] * X[i], args.sr, args.n_fft, args.n_hop)[:track_len]


        # Save track results.
        estimates = {
            'vocals': s[0].T,
            'accompaniment': s[1].T
        }
        scores = museval.eval_mus_track(
            track, estimates, output_dir=args.out_dir
        )
        results.add_track(scores)

    # Save final results in pandas dataframe.
    print(results)
    method = museval.MethodStore()
    method.add_evalstore(results, 'sep_results')
    method_path = out_dir / 'sep_results.pandas'
    method.save(method_path)
