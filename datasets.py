import musdb
import torch
import scipy.signal as ss
import numpy as np
from nmf import *


class Numbers(torch.utils.data.Dataset):
    """ Dummy Dataset """
    def __init__(self, N):
        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, ix):
        x = torch.zeros(self.N)
        x[ix] = 1
        y = torch.tensor(ix)

        return x, y


def stft(x, sr=44100, n_fft=2048, n_hop=512):
    return ss.stft(x, sr,
                   nperseg=n_fft,
                   noverlap=n_fft - n_hop,
                   nfft=n_fft)


class MUSDBDataset(torch.utils.data.Dataset):
    """ Use to iterate over tracks of a single target.
        Use musdb_kwargs to set any necessary musdb properties.
    """

    def __init__(self, musdb_root, is_wav, target,
                 n_fft, n_hop, sr, n_stacked_frames, dtype, device, spgram_pow = 1, silence_thres_db=None,
                 transform=True, **musdb_kwargs):
        self.mus = musdb.DB(musdb_root, is_wav=is_wav, **musdb_kwargs)
        self.target = target
        self.transform = transform
        self.n_fft, self.n_hop, self.sr, self.n_stacked_frames, \
        self.dtype, self.device, self.spgram_pow, self.silence_thres_db = n_fft, n_hop, sr, n_stacked_frames,\
                                                                          dtype, device, spgram_pow,\
                                                                          silence_thres_db

    def __len__(self):
        return len(self.mus)

    def __getitem__(self, ix):
        x = self.mus[ix].targets[self.target].audio.T
        if self.transform:
            x = self.preprocess_input(x, self.n_fft, self.n_hop, self.sr, self.n_stacked_frames,
                                      self.dtype, self.device,
                                      self.spgram_pow, self.silence_thres_db)

        return x

    def preprocess_input(self, x, n_fft, n_hop, sr, n_stacked_frames, dtype, device,
                         spgram_pow=1, silence_thres_db=None):
        # Get STFT.
        t, f, X = stft(x, sr, n_fft, n_hop)
        Xmag = np.mean(np.abs(X), axis=0)

        # Remove silence.
        Px = Xmag ** 2
        phan = np.sum(np.hanning(n_fft))
        rms = np.sqrt((phan / n_fft) ** 2 * (2 * np.sum(Px, axis=0) - Px[0, :]))
        if silence_thres_db is not None:
            ix = 20 * np.log10(rms + 1e-12) > silence_thres_db
            X = Xmag[:, ix] ** spgram_pow
        else:
            X = Xmag ** spgram_pow

        # Stack frames
        X = stack_frames(X, n_stacked_frames)

        return X


class MUSDBTestDataset(torch.utils.data.Dataset):
    """ Use to iterate over tracks of a single target.
        Use musdb_kwargs to set any necessary musdb properties.
    """

    def __init__(self, musdb_root, is_wav,
                 n_fft, n_hop, sr, n_stacked_frames, dtype, device, spgram_pow=1,
                 transform=True, **musdb_kwargs):
        self.mus = musdb.DB(musdb_root, is_wav=is_wav, **musdb_kwargs)
        self.target = 'linear_mixture'
        self.transform = transform
        self.n_fft, self.n_hop, self.sr, self.n_stacked_frames, \
        self.dtype, self.device, self.spgram_pow = n_fft, n_hop, sr, n_stacked_frames,\
                                                                          dtype, device, spgram_pow

    def __len__(self):
        return len(self.mus)

    def __getitem__(self, ix):
        x = self.mus[ix].targets[self.target].audio.T
        vox = self.mus[ix].targets['vocals'].audio.T
        acc = self.mus[ix].targets['accompaniment'].audio.T
        if self.transform:
            X = self.preprocess_input(x, self.n_fft, self.n_hop, self.sr, self.n_stacked_frames,
                                      self.dtype, self.device,
                                      self.spgram_pow)

        return X, x, vox, acc

    def preprocess_input(self, x, n_fft, n_hop, sr, n_stacked_frames, dtype, device,
                         spgram_pow=1):
        # Get STFT.
        t, f, X = stft(x, sr, n_fft, n_hop)

        X = np.abs(X) ** spgram_pow

        # Stack frames
        stacked = []
        for i in range(X.shape[0]):
            stacked.append(stack_frames(X[i], n_stacked_frames))
        X = np.array(stacked)

        return X


if __name__=='__main__':
    # Parameters
    params = {'batch_size': 2,
              'shuffle': True,
              'num_workers': 5}

    dataset = Numbers(6)
    dataset_gen = torch.utils.data.DataLoader(dataset, **params)
    for x, y in dataset_gen:
        print(x, y)
