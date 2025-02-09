# denoiset
DenoisET implements the Noise2Noise algorithm [1] for cryoET data. While this machine learning algorithm has previously been implemented in software such as Cryo-CARE [2] and Topaz-Denoise [3], DenoisET is designed to streamline integration with AreTomo3 and provide a more automated approach that algorithmically decides which tomograms to select for training data and when to transition from training to inference. DenoisET can be run during live data collection, either by applying a previously-trained model for inference or training a new 3D denoising model from scratch. We recommend using the CTF-deconvolved tomograms reconstructed by AreTomo3 for both training and inference. More details are available in Ref. [4].

## Installation

The following will clone DenoisET from Github and install it in a new conda environment.
```console
git clone https://github.com/apeck12/denoiset.git
cd denoiset

conda create --name denoiset python=3.11.4
conda activate denoiset

pip install .
```

## Instructions for use



## Pre-trained models
The following pre-trained models that were trained on data collected at CZII can be found in the models directory:
- cilia.pth: data from mouse olfactory neuronal cilia
- lysosome.pth: data are from affinity-purified endo/lysosomes
- minicell.pth: data are available on the cryoET Data Portal under deposition ID CZCDP-10312
- synaptosome.pth: data are available on the cryoET Data Portal under deposition ID CZCDP-10313

## References
[1] Lehtinen, J. et al. (2018) Noise2Noise: Learning Image Restoration without Clean Data. arXiv: 1803.04189.
[2] Buccholz, T., Jordan, M., Pigino, G. and Jug, F. (2018) Cryo-CARE: Content-Aware Image Restoration for Cryo-Transmission Electron Microscopy Data. arXiv:1810.05420.
[3] Bepler, T., Kelley, K., Noble, A. J, and Berger, B. (2020) Topaz-Denoise: general deep denoising models for cryoEM and cryoET. Nature Communications: 11, 5208.
[4] Peck, A. et al.
