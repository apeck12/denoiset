from argparse import ArgumentParser
import denoiset.inference as inference


def parse_args():
    """Parser for command line arguments."""
    parser = ArgumentParser(
        description="Apply a pre-trained model to denoise tomograms.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Pre-trained UNet3d model file",
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Input directory containing tomograms",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        required=False,
        default="*Vol.mrc",
        help="Glob pattern for file basename",
    )
    parser.add_argument(
        "--filenames",
        type=str,
        required=False,
        help="Text file of specific file basenames",
    )
    parser.add_argument(
        "--exclude_tags",
        type=str,
        required=False,
        nargs="+",
        default=["ODD","EVN"],
        help="Volumes containing these substring(s) will not be denoised",
    )
    parser.add_argument(
        "--length",
        type=int,
        required=False,
        default=128,
        help="Side length of cubic subvolumes to extract in pixels",
    )
    parser.add_argument(
        "--padding",
        type=int,
        required=False,
        default=24,
        help="Padding length in pixels",
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


def main():
    
    config = parse_args()
    if not config.live:
        config.t_interval = config.t_exit = 0
        
    n2n = inference.Denoiser3d(
        config.model,
        config.out_dir,
        config.length,
        config.padding,
    )
    n2n.process(
        config.in_dir,
        pattern=config.pattern,
        exclude_tags=config.exclude_tags,
        filenames=config.filenames,
        t_interval=config.t_interval,
        t_exit=config.t_exit,
    )


if __name__ == "__main__":
    main()
