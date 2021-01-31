# Music Source Separation (undergrad. thesis)
This repo contains the pytorch implementations of MDensenet [1] and Sparse-NMF [2] that were used
for the experiments of my undergraduate thesis. These models can be used in a supervised manner
to learn from clean vocals and accompaniment data in order to separate these sources at test time.
Also, there is a pytorch implementation for the NNDSVD method [4] of initializing the NMF matrices.

The structure of the repo and setup instructions are presented next. Finally, some notes about the 
experiments are given.

## Structure
The repo contains the source relevant to the models as well as setup files.
In the root directory there is `environment-gpu-cuda10.yml` that can be used to create a conda 
environment with the necesesary modules. This configuration file is based on the one used by
Open-Unmix.

In `mdensenet/` there is the MDensenet model's source, a script to separate a track using a trained model and 
an exemplar json file with mdensenet's parameters. The directory contains the
 [efficient_densenet_pytorch]() repo as a submodule, because it is used as a dependency.
 
In `sparse-nmf/` there is the Sparse-NMF implementation along with scripts for training a source dictionary with
[MUSDB18]()'s [3] tracks and using trained dictionaries for the separation of mixtures. 

## Setup instructions
In order to setup the necessary dependencies it is recommended to update the densenet repo submodule
and use anaconda to create a dedicated environment with the dependencies listed in `environment-gpu-cuda10.yml`.
For these, you can run the following commands from your CLI:
1. `conda env create -f environment-gpu-cuda10.yml`
2. `git submodule update --init --recursive`

To activate the dedicated environment, use `conda activate mss-thesis-pytorch-gpu`.

(The conda env setup is based on the Open-Unmix repo, that shows this handy way of installing the necessary
dependencies :pray:)

## Notes on the experiments
The [MUSDB18]() dataset [3] was used along with source from the [Open-Unmix]() repo [5] for the experiments,
in order to train the models and evaluate them on the separation quality as well as on train and 
separation times.

## References
[1] Takahashi, Naoya, and Yuki Mitsufuji. "Multi-scale multi-band densenets for audio source separation." 2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2017.

[2] Le Roux, Jonathan, Felix J. Weninger, and John R. Hershey. "Sparse NMF–half-baked or well done?." Mitsubishi Electric Research Labs (MERL), Cambridge, MA, USA, Tech. Rep., no. TR2015-023 11 (2015): 13-15.

[3] Rafii, Zafar, et al. "MUSDB18-a corpus for music separation." (2017).

[4] Boutsidis, Christos, and Efstratios Gallopoulos. "SVD based initialization: A head start for nonnegative matrix factorization." Pattern recognition 41.4 (2008): 1350-1362.

[5] Stöter, Fabian-Robert, et al. "Open-unmix-a reference implementation for music source separation." Journal of Open Source Software 4.41 (2019): 1667.