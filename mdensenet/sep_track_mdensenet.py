import json
import argparse
import norbert
import torch
import mdensenet as mdn
import numpy as np
from pathlib import Path
import scipy.signal as ss
import librosa
import math
import time
import tqdm
from scipy.io import wavfile


def stft(x, sr, n_fft, n_hop):
    _, _, X = ss.stft(
        x,
        sr,
        nperseg=n_fft,
        noverlap=n_fft - n_hop
    )
    return X


def istft(X, sr, n_fft, n_hop):
    _, x = ss.istft(
        X / (n_fft / 2),
        sr,
        nperseg=n_fft,
        noverlap=n_fft - n_hop,
        boundary=True
    )
    return x


parser = argparse.ArgumentParser(description='Separate a mixture to vocals and accompaniment using a trained mdensenet '
                                             'model.')
parser.add_argument('--track', type=str, help='Specify the input mixture track.',
                    required=True)
parser.add_argument('--model-vox', type=str, help='Path to the torch .pth model trained on vocals sources.',
                    required=True)
parser.add_argument('--params-json', type=str, help='Path to the .json file containing the model parameters.',
                    required=True)
parser.add_argument('--out-name', type=str, help='Specify the output path for the track. Use a name without the '
                                                 'extension eg. "mydir/mytrack" .',
                    required=True, default='track')
parser.add_argument('--n-fft', type=int, help='FFT length for the stft.',
                    required=False,
                    default=4096)
parser.add_argument('--n-hop', type=int, help='Hop length for the stft.',
                    required=False,
                    default=1024)
parser.add_argument('--segment-time', type=float, help='Segment time in sec. Used to split the input spectrogram and '
                                                       'process it piece by piece to reduce memory requirements.',
                    required=False,
                    default=3.0)
parser.add_argument('--device', type=str, help='Select the device that does the processing.',
                    required=False, choices=['cpu', 'cuda'],
                    default='cuda')


args, _ = parser.parse_known_args()

device = torch.device(args.device)

# Init model using params .json file.
params_path = Path(args.params_json).expanduser()
with open(params_path, 'r') as stream:
    model_params = json.load(stream)
model = mdn.MDenseNet(**model_params)

# Load model from .pth file.
model_path = Path(args.model_vox).expanduser()
state = torch.load(
    model_path,
    map_location=device
)
model.load_state_dict(state)
model.eval()
model.to(device)
model.mdensenet.stft.center = True

# Prepare for processing.
n_fft = args.n_fft
n_hop = args.n_hop
x, sr = librosa.load(Path(args.track).expanduser(), sr=None, mono=False)

start_time = time.time()

V = []  # will hold both sources spectrograms.
audio = torch.tensor(x[None, ...]).to(device)
with torch.no_grad():
    x = model.mdensenet.transform(audio)  # perform stft
    num_frames = math.ceil((args.segment_time * sr - n_fft) / n_hop)
    X = torch.split(x, num_frames, dim=3)
    Vj = []  # holds vocals' spectrograms.
    for i in tqdm.tqdm(range(len(X)), desc='Estimating vocals..'):
        Vj.append(model(X[i]))
    Vj = torch.cat(Vj, dim=3).cpu().detach().numpy()

# Prepare input for MWF.
print('Calculating MWF..')
V_vox = np.transpose(Vj, [3, 0, 1, 2])
V.append(V_vox[:, 0, ...])  # remove sample dim
V = np.transpose(np.array(V), (1, 3, 2, 0))

X = model.mdensenet.stft(audio).detach().cpu().numpy()
X = X[..., 0] + X[..., 1] * 1j
X = X[0].transpose(2, 1, 0)
V = norbert.residual_model(V, X, 1)
Y = norbert.wiener(V, X.astype(np.complex128), 1,
                   use_softmask=False)

# Extract source estimates in time domain.
s = []
estimates = {}
for j in range(Y.shape[-1]):
    audio_hat = istft(
        Y[..., j].T,
        n_fft=n_fft,
        n_hop=n_hop,
        sr=sr
    )
    s.append(audio_hat.T)

end_time = time.time()
print(f'Separation duration: {end_time - start_time:.2f} sec.')

print('Saving track..')
out_name = Path(args.out_name).expanduser()
out_name.parent.mkdir(parents=True, exist_ok=True)

wavfile.write(str(out_name) + '_vox.wav', sr, s[0].astype(np.float32))
wavfile.write(str(out_name) + '_acc.wav', sr, s[1].astype(np.float32))
