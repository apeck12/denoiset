import numpy as np
import mrcfile
import glob
import os


def load_mrc(filename: str) -> np.ndarray:
    """
    Load the data in an mrc file into a numpy array.

    Parameters
    ----------
    filename: str, path to mrc file

    Returns
    -------
    np.ndarray, image or volume
    """
    with mrcfile.open(filename, "r", permissive=True) as mrc:
        return mrc.data


def save_mrc(
    data: np.ndarray,
    filename: str,
    overwrite: bool = True,
    apix: float = None,
):
    """
    Save a numpy array to mrc format.

    Parameters
    ----------
    data: np.ndarray, image or volume
    filename: str, save path
    overwrite: bool, overwrite filename if already exists
    apix: float, pixel size in Angstrom
    """
    if data.dtype != np.dtype("float32"):
        data = data.astype(np.float32)
    with mrcfile.new(filename, overwrite=overwrite) as mrc:
        mrc.set_data(data)
        if apix:
            mrc.voxel_size = apix


def get_voxel_size(
    filename: str,
    isotropic: bool = True,
) -> float:
    """
    Extract voxel size from mrc file.

    Parameters
    ----------
    filename: str, path to mrc file
    isotropic: bool, extract single value assuming isotropic pixel size

    Returns
    -------
    apix: float, pixel size in Angstrom
    """
    apix = mrcfile.open(filename).voxel_size.tolist()
    if isotropic:
        return apix[0]
    return apix


def expand_filelist(
    in_dir: str,
    pattern: str,
    exclude_tags: list=[],
) -> list[str]:
    """
    Retrieve a list of files in in_dir whose basename contains
    the specified glob-expandable pattern, excluding any files
    that contain the specified exclusion strings.
    
    Parameters
    ----------
    in_dir: base directory 
    pattern: glob-expandable string pattern
    exclude_tags: list of tags to exclude
    
    Returns
    -------
    filenames: filenames that match search
    """
    filenames = glob.glob(os.path.join(in_dir, pattern))
    for tag in exclude_tags:
        filenames = [fn for fn in filenames if tag not in fn]
    return filenames


def get_split_filenames(
    in_dir: str, 
    pattern: str, 
    f_val: float,
    exclude_tags: list=[], 
    rng: np.random._generator.Generator=None,
) -> dict:
    """
    Retrieve all available pairs of files and split between train
    and test sets.
    
    Parameters
    ----------
    in_dir: base directory 
    pattern: glob-expandable string pattern
    fval: fraction of files to set aside for test set
    exclude_tags: list of tags to exclude
    rng: random generator object if seed is fixed
    
    Returns
    -------
    dict: filenames for train1, train2, valid1, and valid2 sets
    """
    if rng is None:
        rng = np.random.default_rng()
        
    filenames1 = expand_filelist(in_dir, pattern)
    if len(filenames1) == 0:
        raise IOError("No input ODD files found")

    filenames2 = [fn1.replace("ODD", "EVN") for fn1 in filenames1]
    keep_idx = [os.path.exists(fn2) for fn2 in filenames2]
    if len(keep_idx) != len(filenames2):
        print("Warning! Not every ODD file has a corresponding EVN file")
        filenames1 = list(np.take(filenames1, keep_idx))
        filenames2 = list(np.take(filenames2, keep_idx))

    assert 0 < f_val < 1
    num_val = int(np.around(f_val * len(filenames1)))
    rand_idx = rng.choice(len(filenames1), size=num_val, replace=False)
    
    file_split = {}
    file_split['train1'] = [fn for i,fn in enumerate(filenames1) if i not in rand_idx]
    file_split['train2'] = [fn for i,fn in enumerate(filenames2) if i not in rand_idx]
    file_split['test1'] = list(np.take(filenames1, rand_idx))
    file_split['test2'] = list(np.take(filenames2, rand_idx))

    return file_split
