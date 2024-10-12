import os
from argparse import ArgumentParser
import denoiset.training as training
from denoiset.settings import ProcessingConfigTrain3d


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

    return parser.parse_args()


def store_parameters(config):
    """
    Store command line arguments in a json file.
    """
    d_config = vars(config)

    reconfig = {}
    reconfig["software"] = {"name": "denoiset", "version": "0.1.0"}
    reconfig["input"] = {k: d_config[k] for k in ["in_path", "model"]}
    reconfig["output"] = {k: d_config[k] for k in ["out_dir"]}

    used_keys = [list(reconfig[key].keys()) for key in reconfig]
    used_keys = [p for param in used_keys for p in param]
    param_keys = [key for key in d_config if key not in used_keys]
    reconfig["parameters"] = {k: d_config[k] for k in param_keys}

    reconfig = ProcessingConfigTrain3d(**reconfig)

    os.makedirs(config.out_dir, exist_ok=True)
    with open(os.path.join(config.out_dir, "train3d.json"), "w") as f:
        f.write(reconfig.model_dump_json(indent=4))


def main():
    
    config = parse_args()
    if not config.live:
        config.t_interval = config.t_exit = 0
    else:
        raise NotImplementedError
    store_parameters(config)
        
    n2n = training.Trainer3d(
        config.in_path,
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

