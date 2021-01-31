import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as Fnc
import sys

sys.path.append('./efficient_densenet_pytorch/models')
from densenet import _DenseLayer, _Transition, _DenseBlock


class _UpSampleLayer(nn.Sequential):
    r""" 
    UpSampleLayer is consisted of a batch normalization, a relu and
    a transposed convolution layer. Make sure that the transposed convolution
    has the same kernel size as the pooling layer.
    """

    def __init__(self, num_input_features, num_output_features):
        super(_UpSampleLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('tconv', nn.ConvTranspose2d(num_input_features, num_output_features,
                                                    kernel_size=2, stride=2))


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        center=False
    ):
        super(STFT, self).__init__()
        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )
        return stft_f


class Spectrogram(nn.Module):
    def __init__(
        self,
        power=1,
        mono=True
    ):
        super(Spectrogram, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_samples, nb_channels, nb_bins, nb_frames)
        """
        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)

        return stft_f


class _MDenseNet(nn.Module):
    """
    This class implements the core of the mdensenet model.
    :param num_init_features is the number of feature maps calculated by the input convolutional layer.
    :param growth_rate is the number of feature maps calculated by each dense layer.
    :param block_config is a list of ints, containing the number of dense layers that each dense block contains.
    :param compression is a value in (0, 1] showing the percentage of feature maps produced by the down/up sampling
           layers.
    :param bn_size is a multiplier for the amount of feature maps produced by each bottleneck layer.
            #maps = bn_size * growth_rate
    :param drop_rate is the droupout rate for a neuron, with 0 meaning no dropout is used.
    :param efficient when True the computations are memory efficient with the expense of some time.
    :param n_fft is the FFT length for the STFT.
    :param n_hop is the hop length for the STFT.
    :param power is the spectrogram power.
    :param mono if set to True, the input will be expected to be single channel.
    :param input_mean is an ndarray of size (nb_bins,) with the mean across time frames. If left none, it is 0 vector.
    :param input_scale is an ndarray of size (nb_bins,) with the standard deviation across time frames. If left none
            it is a vector of 1s.
    :param max_bin is the maximum frequency bin used for processing.
    """
    def __init__(self, num_init_features=32, growth_rate=4, block_config=(14, 14, 14), compression=0.5,
                 bn_size=4, drop_rate=0, efficient=True, n_fft=4096, n_hop=1024, power=1, mono=False,
                 input_mean=None, input_scale=None, max_bin=None):

        super(_MDenseNet, self).__init__()

        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=mono)

        self.transform = nn.Sequential(self.stft, self.spec)

        self.total_scales = len(block_config)
        self.n_transitions = (self.total_scales - 1)

        self.features = nn.Sequential(OrderedDict([
            ('zeropad0', nn.ZeroPad2d((1, 1, 1, 2))),
            ('conv0', nn.Conv2d(2, num_init_features, kernel_size=(4, 3), bias=False))
        ]))

        self.nb_bins = n_fft // 2 + 1

        if max_bin is not None:
            self.nb_bins = max_bin

        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:self.nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:self.nb_bins]
            ).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = nn.Parameter(input_mean.reshape([1, 1, self.nb_bins, 1]), requires_grad=False)
        self.input_scale = nn.Parameter(input_scale.reshape([1, 1, self.nb_bins, 1]), requires_grad=False)

        feature_stack = []  # keeps the skip connection num of features.
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )

            self.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i < self.total_scales - 1:
                feature_stack.append(num_features)

                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        for i, num_layers in enumerate(reversed(block_config[:-1])):
            upsample = _UpSampleLayer(num_input_features=num_features,
                                      num_output_features=int(num_features * compression))
            self.add_module('upsample%d' % (self.total_scales + i + 1), upsample)

            # Add the skip connection features to the input
            num_features_skip_connection = feature_stack.pop()
            num_features = int(num_features * compression) + num_features_skip_connection

            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denseblock%d' % (self.total_scales + i + 1), block)
            num_features += num_layers * growth_rate

        self.num_output_channels = num_features

    def forward(self, x):
        x = x[:, :, :self.nb_bins, :]
        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.input_mean
        x *= self.input_scale

        B, C, F, N = x.shape
        frequency_pad = (2 ** self.n_transitions - (F % 2 ** self.n_transitions)) % 2 ** self.n_transitions
        time_frames_pad = (2 ** self.n_transitions - (N % 2 ** self.n_transitions)) % 2 ** self.n_transitions
        x = Fnc.pad(x, (0, time_frames_pad,
                      0, frequency_pad))

        feature_stack = []

        # initial layer
        out = self.features(x)

        # downsampling path
        for i in range(self.total_scales - 1):
            layer = getattr(self, f'denseblock{i + 1}')
            out = layer(out)
            feature_stack.append(out)

            layer = getattr(self, f'transition{i + 1}')
            out = layer(out)

        # middle layer
        layer = getattr(self, f'denseblock{self.total_scales}')
        out = layer(out)

        # upsampling path
        for i in range(self.total_scales - 1):
            layer = getattr(self, f'upsample{self.total_scales + i + 1}')
            out = torch.cat([layer(out), feature_stack.pop()], dim=1)
            layer = getattr(self, f'denseblock{self.total_scales + i + 1}')
            out = layer(out)

        return out[:, :, :F, :N]


class MDenseNet(nn.Sequential):
    def __init__(self, num_init_features=32, growth_rate=4, block_config=(14, 14, 14), compression=1.,
                 bn_size=4, drop_rate=0, efficient=True, final_growth_rate=4, num_final_layers=2, num_output_channels=2,
                 input_mean=None, input_scale=None, max_bin=None):
        """
        This class implements an mdensenet model based on
        Takahashi, Naoya, and Yuki Mitsufuji. "Multi-scale multi-band densenets for audio source separation."
        2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2017.

        :param num_init_features is the number of feature maps calculated by the input convolutional layer.
        :param growth_rate is the number of feature maps calculated by each dense layer.
        :param block_config is a list of ints, containing the number of dense layers that each dense block contains.
        :param compression is a value in (0, 1] showing the percentage of feature maps produced by the down/up sampling
               layers.
        :param bn_size is a multiplier for the amount of feature maps produced by each bottleneck layer.
                #maps = bn_size * growth_rate
        :param drop_rate is the droupout rate for a neuron, with 0 meaning no dropout is used.
        :param efficient when True the computations are memory efficient with the expense of some time.
        :param final_growth_rate is used for the dense block in the output.
        :param num_final_layers is the number of dense layers used in the output dense block.
        :param num_output_channels is the number of output channels.

        :param n_fft is the FFT length for the STFT.
        :param n_hop is the hop length for the STFT.
        :param power is the spectrogram power.
        :param mono if set to True, the input will be expected to be single channel.
        :param input_mean is an ndarray of size (nb_bins,) with the mean across time frames. If left none, it is 0
                vector.
        :param input_scale is an ndarray of size (nb_bins,) with the standard deviation across time frames. If left none
                it is a vector of 1s.
        :param max_bin is the maximum frequency bin used for processing.
        """
        self.max_bin = max_bin
        super(MDenseNet, self).__init__()

        # input & core module
        self.mdensenet = _MDenseNet(num_init_features, growth_rate, block_config, compression,
                                    bn_size, drop_rate,
                                    efficient, input_mean=input_mean, input_scale=input_scale, max_bin=max_bin)
        self.add_module('mdensenet', self.mdensenet)

        # output module
        self.add_module('denseFinal', _DenseBlock(
            num_layers=num_final_layers,
            num_input_features=self.mdensenet.num_output_channels,
            bn_size=bn_size,
            growth_rate=final_growth_rate,
            drop_rate=drop_rate,
            efficient=efficient,
        ))
        num_features = self.mdensenet.num_output_channels + num_final_layers * final_growth_rate
        self.add_module('padFinal',  nn.ZeroPad2d((0, 0, 0, 1)))
        self.add_module('convFinal', nn.Conv2d(num_features, num_output_channels, kernel_size=(2, 1), bias=False))
        self.add_module('reluFinal', nn.ReLU(inplace=True))
