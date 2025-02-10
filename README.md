# denoiset
DenoisET implements the Noise2Noise algorithm [1] for cryoET data. While this machine learning algorithm has previously been implemented in software such as Cryo-CARE [2] and Topaz-Denoise [3], DenoisET is designed to streamline integration with AreTomo3 and provide a more automated approach that algorithmically decides which tomograms to select for training data and when to transition from training to inference. DenoisET can be run during live data collection, either by applying a previously-trained model for inference or training a new 3D denoising model from scratch. We recommend using the CTF-deconvolved tomograms reconstructed by AreTomo3 for both training and inference. More details are available in Ref. [4].

## Installation

The following will clone DenoisET from Github and install it in a new conda environment.
```
git clone https://github.com/apeck12/denoiset.git
cd denoiset

conda create --name denoiset python=3.11.4
conda activate denoiset

pip install .
```

## Instructions for use

The following command runs both training and inference, with training tomograms selected based on their quality metrics and the transition to inference automatically determined based on the appearance of checkerboard artifacts in the denoised tomograms:
```
denoise3d --input [tomogram directory] --output [output directory] --metrics_file [path to TiltSeries_Metrics.csv] 
```
By default, the file nomenclature is expected to follow AreTomo3's conventions, with paired half tomograms ending in ODD_Vol.mrc and EVN_Vol.mrc. Only tomograms without EVN or ODD in their filenames will be denoised during inference. All tomograms (even, odd and full) are expected to be in the same input directory. A `training` subdirectory will be created in the output directory and populated with the list of training tomograms, the per-epoch model files(epoch[n].pth), per-epoch statistics (training_stats.csv), and representative denoised tomograms during each training epoch.

To adjust the volume of training data, the `--n_extract` (default: 250) and `--max_selected` (default: 30) parameters respectively determine the number of subvolumes extracted per tomogram per training epoch and the maximum number of tomograms to use during training. The default minimum number of tomograms (`--min_selected`) is 10. The default number of training epochs is 20, but training will automatically be terminated and inference started once the threshold set by `--ch_threshold` (default 0.034) is reached. 

To run this concurrently with AreTomo3 processing, the `--live` flag can be supplied. The `--t_interval` and `--t_exit` flags determine how frequently DenoisET checks for new tomograms and how long it waits to exit after finding no new tomograms. The default values for these are 5 and 30 minutes, respectively, but the parameters should be given in seconds.

The following flags can be used to adjust the default thresholds for the quality metrics:
* `--tilt_axis`, maximum deviation from the median value in degrees (default: 1.5)
* `--thickness`, minimal sample thickness in Angstrom (default: 1000)
* `--global_shift`, maximum global shift in Angstrom (default: 750)
* `--bad_patch_low`, maximum fraction of bad patches at low tilt angles (default: 0.05)
* `--bad_patch_all`, maximum fraction of bad patches across the tilt range (default: 0.1)
* `--ctf_res`, maximum resolution of the CTF score in Angstrom (default: 20)
* `--ctf_score`, minimum CTF score (default: 0.05)

Running `denoise3d --help` will list dditional parameters that can be adjusted from the command line.

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
