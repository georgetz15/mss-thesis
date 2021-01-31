import argparse
from pathlib import Path
import librosa
import torch
import tqdm
import scipy.signal as ss
import numpy as np
from scipy.io import wavfile
import time

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

    parser = argparse.ArgumentParser(description='Tool to separate a mixture audio file into vocals and accompaniment '
                                                 'using the Sparse NMF model.')

    parser.add_argument('--track', type=str, help='Path to audio file.',
                        required=True)
    parser.add_argument('--acc', type=str, help='Path to accompaniment model',
                        required=True)
    parser.add_argument('--vox', type=str, help='Path to vocals model',
                        required=True)

    parser.add_argument('--out-name', type=str, help='Path for output tracks with name.',
                        default='out')
    parser.add_argument('--sr', type=float, default=44100,
                        help='Sampling rate of the tracks.')
    parser.add_argument('--n-fft', type=int, default=2048)
    parser.add_argument('--n-hop', type=int, default=512)
    parser.add_argument('--tol', type=float, default=1e-5, help='Cost change tolerance for stopping criterion.')
    parser.add_argument('--n-stacked-frames', type=int, default=6, help='# of stacked frames.')
    parser.add_argument('--sparsity-weight', type=float, default=0,
                        help='regularization weight, defaults to 0')
    parser.add_argument('--b-div', type=float, default=1, help='Beta divergence to be used.')
    parser.add_argument('--specgram-pow', type=float, default=1,
                        help='Power of abs(stft(x))')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='Choose device for training')

    args, _ = parser.parse_known_args()

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

    track = Path(args.track).expanduser()
    x, sr = librosa.load(str(track), sr=args.sr, mono=False)
    if x.ndim == 1:
        x = x[None, ...]
    n_sources = 2

    start_time = time.time()
    n_channels, n_samples = x.shape
    X = stft(x, args.sr, args.n_fft, args.n_hop)[2]

    # Preprocessing
    V = np.abs(X) ** args.specgram_pow
    # Stack frames
    stacked = []
    for i in range(n_channels):
        stacked.append(stack_frames(V[i], args.n_stacked_frames))
    V = torch.tensor(stacked, device=args.device, dtype=model.dtype)  # input spectrograms/supervectors.
    s = np.zeros((n_sources, n_channels, n_samples), dtype=np.float32)  # output time-domain sources.
    for i in tqdm(range(n_channels), desc='Separating channel'):
        H = model.infer(V[i] + 1e-12)
        Vvox = (model.W[:, vox_ix] @ H[vox_ix, :])
        Vacc = (model.W[:, acc_ix] @ H[acc_ix, :])
        D = Vvox + Vacc
        Gvox = (Vvox / D).cpu().numpy()
        Gvox = unstack_frames(Gvox, args.n_stacked_frames)
        Gacc = (Vacc / D).cpu().numpy()
        Gacc = unstack_frames(Gacc, args.n_stacked_frames)
        s[0, i] = istft(Gvox[:, :X.shape[-1]] * X[i], args.sr, args.n_fft, args.n_hop)[:n_samples]
        s[1, i] = istft(Gacc[:, :X.shape[-1]] * X[i], args.sr, args.n_fft, args.n_hop)[:n_samples]

    end_time = time.time()
    print(f'Separation duration: {end_time - start_time:.2f} sec.')

    out_name = Path(args.out_name).expanduser()
    wavfile.write(str(out_name) + '_vox.wav', rate=args.sr, data=s[0].T)
    wavfile.write(str(out_name) + '_acc.wav', rate=args.sr, data=s[1].T)
