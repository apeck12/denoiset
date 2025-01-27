import os
import time
from argparse import ArgumentParser
import denoiset.training as training
import denoiset.curation as curation
from denoiset.settings import ProcessingConfigTrain3d
from denoiset.args import AttrDict


def parse_args():
    """Parser for command line arguments."""
    parser = ArgumentParser(
        description="Train a denoising model using the Noise2Noise framework",
    )
    parser.add_argument(
        "--in_path",
        type=str,
        required=True,
        help="Directory or text file (listing path and basename) of training tomograms",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Pre-trained UNet3d model file",
    )
    parser.add_argument(
        "--n_extract",
        type=int,
        required=False,
        default=100,
        help="Number of subvolumes to extract per tomogram",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Fixed random seed value",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        default="adagrad",
        help="Optimizer",    
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=8,
        help="Number of paired subvolumes per batch",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        required=False,
        default=0.1,
        help="Fraction of tomograms for validation",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        required=False,
        default="*ODD_Vol.mrc",
        help="Glob pattern for ODD tomograms",
    )
    parser.add_argument(
        "--extension",
        type=str,
        required=False,
        default="_ODD_Vol.mrc",
        help="suffix for ODD tomograms",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        required=False,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--n_denoise",
        type=int,
        required=False,
        default=3,
        help="Number of tomograms to denoise per epoch for visual inspection",
    )
    parser.add_argument(
        "--length",
        type=int,
        required=False,
        default=96,
        help="Side length of cubic subvolumes to extract in pixels",
    )
    parser.add_argument(
        "--live",
        required=False,
        action="store_true",
        help="Live processing mode to denoise tomograms on-the-fly",
    )
    parser.add_argument(
        "--t_interval",
        required=False,
        type=float,
        default=300,
        help="Interval in seconds between checking for new files",
    )
    parser.add_argument(
        "--t_exit",
        required=False,
        type=float,
        default=1800,
        help="Exit after this period in seconds if new files are not found",
    )
    parser.add_argument(
        "--min_selected",
        required=False,
        type=int,
        default=10,
        help="Minimum number of selected tomograms if in_path is a metrics file",
    )
    parser.add_argument(
        "--max_selected",
        required=False,
        type=int,
        default=50,
        help="Maximum number of selected tomograms if in_path is a metrics file",
    )
    parser.add_argument(
        "--sort_by",
        required=False,
        type=str,
        default="Global_Shift",
        help="Metric for sorting tomograms if selected set exceeds max_selected",
    )
    parser.add_argument(
        "--vol_path",
        required=False,
        type=str,
        help="Path to volumes to denoise if not in the same directory as the metrics file",
    )
    parser.add_argument(
        "--tilt_axis",
        required=False,
        type=float,
        default=1.5,
        help="Maximum deviation from median tilt axis in degrees",
    )
    parser.add_argument(
        "--thickness",
        required=False,
        type=float,
        default=1000,
        help="Minimum sample thickness in Angstrom",
    )
    parser.add_argument(
        "--global_shift",
        required=False,
        type=float,
        default=750,
        help="Maximum global shift in Angstrom",
    )
    parser.add_argument(
        "--bad_patch_low",
        required=False,
        type=float,
        default=0.05,
        help="Maximum fraction of bad patches at low tilt angles",
    )
    parser.add_argument(
        "--bad_patch_all",
        required=False,
        type=float,
        default=0.1,
        help="Maximum fraction of bad patches across the full tilt range",
    )
    parser.add_argument(
        "--ctf_res",
        required=False,
        type=float,
        default=20.0,
        help="Maximum resolution of CTF score in Angstrom",
    )
    parser.add_argument(
        "--ctf_score",
        required=False,
        type=float,
        default=0.05,
        help="Minimum CTF score",
    )

    return parser.parse_args()


def store_parameters(config):
    """
    Store command line arguments in a json file.
    """
    reconfig = {}
    reconfig["software"] = {"name": "denoiset", "version": "0.1.0"}
    reconfig["input"] = {k: config[k] for k in ["in_path", "model", "vol_path"]}
    reconfig["output"] = {k: config[k] for k in ["out_dir"]}

    used_keys = [list(reconfig[key].keys()) for key in reconfig]
    used_keys = [p for param in used_keys for p in param]
    param_keys = [key for key in config if key not in used_keys]
    reconfig["parameters"] = {k: config[k] for k in param_keys}

    reconfig = ProcessingConfigTrain3d(**reconfig)
    os.makedirs(config.out_dir, exist_ok=True)
    with open(os.path.join(config.out_dir, "train3d.json"), "w") as f:
        f.write(reconfig.model_dump_json(indent=4))


def main():
    
    config = parse_args()
    config = AttrDict(vars(config))
    os.makedirs(config.out_dir, exist_ok=True)
    
    if not config.live:
        config.t_interval = config.t_exit = 0

    if os.path.basename(config.in_path) == "TiltSeries_Metrics.csv":
        curator = curation.TomogramCurator(config.in_path)
        metrics = ['tilt_axis', 'thickness', 'global_shift',
                   'bad_patch_low', 'bad_patch_all', 'ctf_res', 'ctf_score']        
        for m in metrics:
            curator.reset_criterion(m.title().replace('Ctf', 'CTF'), config[m])
        
        in_path = os.path.join(config.out_dir, "traininglist.txt")
        if config.vol_path is None:
            config.vol_path = os.path.dirname(config.in_path)
        curator.curate_live(
            out_file=in_path,
            vol_path=config.vol_path,
            max_selected=config.max_selected,
            min_selected=config.min_selected,
            sort_by=config.sort_by,
            t_interval=config.t_interval,
            t_exit=config.t_exit,
        )
        time.sleep(config.t_interval)
        
    else:
        # TO-DO: handle case where in_path is provided and only basenames are in text file
        in_path = config.in_path
    store_parameters(config)
        
    n2n = training.Trainer3d(
        in_path,
        config.out_dir,
        fn_model=config.model,
        seed=config.seed,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        pattern=config.pattern,
        extension=config.extension,
        length=config.length,
        n_extract=config.n_extract,
    )
    n2n.train(
        n_epochs=config.n_epochs, 
        n_denoise=config.n_denoise,
    )
    
    
if __name__ == "__main__":
    main()

