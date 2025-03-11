import argparse


class BaseArgs:
    """ Class that consolidates arguments for 3d denoising. """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
    def add_curation_args(self):
        """ Arguments related to metrics-based tomogram curation. """
        self.parser.add_argument(
            "--metrics_file",
            required=False,
            type=str,
            help="AreTomo3 TiltSeries_Metrics.csv file",
        )
        self.parser.add_argument(
            "--min_selected",
            required=False,
            type=int,
            default=10,
            help="Minimum number of selected tomograms if in_path is a metrics file",
        )
        self.parser.add_argument(
            "--max_selected",
            required=False,
            type=int,
            default=30,
            help="Maximum number of selected tomograms if in_path is a metrics file",
        )
        self.parser.add_argument(
            "--sort_by",
            required=False,
            type=str,
            default="Global_Shift",
            help="Metric for sorting tomograms if selected set exceeds max_selected",
        )
        self.parser.add_argument(
            "--tilt_axis",
            required=False,
            type=float,
            default=1.5,
            help="Maximum deviation from median tilt axis in degrees",
        )
        self.parser.add_argument(
            "--thickness",
            required=False,
            type=float,
            default=1000,
            help="Minimum sample thickness in Angstrom",
        )
        self.parser.add_argument(
            "--global_shift",
            required=False,
            type=float,
            default=750,
            help="Maximum global shift in Angstrom",
        )
        self.parser.add_argument(
            "--bad_patch_low",
            required=False,
            type=float,
            default=0.05,
            help="Maximum fraction of bad patches at low tilt angles",
        )
        self.parser.add_argument(
            "--bad_patch_all",
            required=False,
            type=float,
            default=0.1,
            help="Maximum fraction of bad patches across the full tilt range",
        )
        self.parser.add_argument(
            "--ctf_res",
            required=False,
            type=float,
            default=20.0,
            help="Maximum resolution of CTF score in Angstrom",
        )
        self.parser.add_argument(
            "--ctf_score",
            required=False,
            type=float,
            default=0.05,
            help="Minimum CTF score",
        )
        
    def add_io_args(self):
        """ Arguments related to I/O. """
        self.parser.add_argument(
            "--input",
            type=str,
            required=True,
            help="Input directory of tomograms or a text file specifying their full path (minus extension)",
        )
        self.parser.add_argument(
            "--model",
            type=str,
            required=False,
            help="Pre-trained UNet3d model file",
        )
        # TO DO: ADD CHECKPOINT STATE DICT
        self.parser.add_argument(
            "--output", 
            type=str,
            required=True,
            help="Output directory for denoised volumes", 
        )
        self.parser.add_argument(
            "--pattern",
            type=str,
            required=False,
            default="*Vol.mrc",
            help="Glob pattern for file basename",
        )

    def add_live_args(self):
        """ Arguments related to live processing. """
        self.parser.add_argument(
            "--live",
            required=False,
            action="store_true",
            help="Live processing mode to denoise tomograms on-the-fly",
        )
        self.parser.add_argument(
            "--t_interval",
            required=False,
            type=float,
            default=300,
            help="Interval in seconds between checking for new files",
        )
        self.parser.add_argument(
            "--t_exit",
            required=False,
            type=float,
            default=1800,
            help="Exit after this period in seconds if new files are not found",
        )

    def add_training_args(self):
        """ Arguments related to training. """
        self.parser.add_argument(
            "--odd_pattern",
            type=str,
            required=False,
            default="*ODD_Vol.mrc",
            help="Glob pattern for ODD tomograms",
        )
        self.parser.add_argument(
            "--odd_extension",
            type=str,
            required=False,
            default="_ODD_Vol.mrc",
            help="suffix for ODD tomograms",
        )
        self.parser.add_argument(
            "--n_extract",
            type=int,
            required=False,
            default=250,
            help="Number of subvolumes to extract per tomogram",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            required=False,
            help="Fixed random seed value",
        )
        self.parser.add_argument(
            "--optimizer",
            type=str,
            required=False,
            default="adagrad",
            help="Optimizer",    
        )
        self.parser.add_argument(
            "--learning_rate",
            type=float,
            required=False,
            default=0.001,
            help="Learning rate",
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            required=False,
            default=8,
            help="Number of paired subvolumes per batch",
        )
        self.parser.add_argument(
            "--val_fraction",
            type=float,
            required=False,
            default=0.1,
            help="Fraction of tomograms for validation",
        )
        self.parser.add_argument(
            "--n_epochs",
            type=int,
            required=False,
            default=20,
            help="Number of training epochs",
        )
        self.parser.add_argument(
            "--n_denoise",
            type=int,
            required=False,
            default=3,
            help="Number of tomograms to denoise per epoch for visual inspection",
        )
        self.parser.add_argument(
            "--length",
            type=int,
            required=False,
            default=96,
            help="Side length of cubic subvolumes to extract in pixels",
        )
        self.parser.add_argument(
            "--train_only",
            required=False,
            action="store_true",
            help="Only perform training and not inference on the full dataset",
        )
        self.parser.add_argument(
            "--ch_threshold",
            type=float,
            required=False,
            default=0.034,
            help="Checkerboard metric threshold for terminating training",
        )
        self.parser.add_argument(
            "--train_all_epochs",
            required=False,
            action="store_true",
            help="Continue training past ch_threshold for diagnostic purposes",
        )
        
    def add_inference_args(self):
        """ Arguments related to inference. """
        self.parser.add_argument(
            "--exclude_tags",
            type=str,
            required=False,
            nargs="+",
            default=["ODD","EVN"],
            help="Volumes containing these substring(s) will not be denoised",
        )
        self.parser.add_argument(
            "--inf_length",
            type=int,
            required=False,
            default=128,
            help="Side length of cubic subvolumes to extract in pixels during inference",
        )
        self.parser.add_argument(
            "--inf_padding",
            type=int,
            required=False,
            default=24,
            help="Padding length in pixels during inference",
        )

    def parse_args(self):
        return self.parser.parse_args()

    
class PredictArgs(BaseArgs):
    def __init__(self):
        super().__init__()
        self.add_io_args()
        self.add_inference_args()
        self.add_live_args()

        
class DenoiseArgs(BaseArgs):
    def __init__(self):
        super().__init__()
        self.add_io_args()
        self.add_curation_args()
        self.add_training_args()
        self.add_inference_args()
        self.add_live_args()


class CurateArgs(BaseArgs):
    def __init__(self):
        super().__init__()
        self.add_curation_args()
        self.parser.add_argument(
            "--output", 
            type=str,
            required=True,
            help="Output CSV file of the selected tilt-series", 
        )

        
class AttrDict(dict):
    """
    A class to convert a nested Dictionary into an object with key-values
    accessible using attribute notation (AttrDict.attribute) in addition to
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse into nested dicts (like: AttrDict.attr.attr)

    Adapted from: https://stackoverflow.com/a/48806603
    """

    def __init__(self, mapping=None, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        if mapping is not None:
            for key, value in mapping.items():
                self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)
        self.__dict__[key] = value  

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)
