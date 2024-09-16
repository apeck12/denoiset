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
    apix: float = None,
    overwrite: bool = True,
):
    """
    Save a numpy array to mrc format.

    Parameters
    ----------
    data: np.ndarray, image or volume
    filename: str, save path
    apix: float, pixel size in Angstrom
    overwrite: bool, overwrite filename if already exists
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
    in_path: str, 
    f_val: float,
    pattern: str="*ODD_Vol.mrc",
    extension: str="_ODD_Vol.mrc",
    exclude_tags: list=[], 
    rng: np.random._generator.Generator=None,
) -> dict:
    """
    Retrieve available file pairs and split into train and validation. 
    Files are supplied via the in_path argument as a 1) directory of
    mrc files, 2) list of *ODD_Vol.mrc file paths, or 3) text file in
    which even line specifies the path to a tomogram -- either the base
    name or omitting the ODD_Vol.mrc extension. 
    
    Parameters
    ----------
    in_path: base directory or text file listing files
    f_val: fraction to set aside for validation set
    pattern: glob-expandable string pattern
    extension: extension to add to each listed file
    exclude_tags: list of tags to exclude
    rng: random generator object if seed is fixed
    
    Returns
    -------
    dict: filenames for train1, train2, valid1, and valid2 sets
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if isinstance(in_path, list):
        filenames1 = in_path
    elif os.path.isdir(in_path):
        filenames1 = expand_filelist(in_path, pattern)
    elif os.path.isfile and os.path.splitext(in_path)[-1] != '.mrc':
        filenames1 = np.loadtxt(in_path, dtype=str)
        if not all([os.path.splitext(fn)[-1]=='.mrc' for fn in filenames1]):
            filenames1 = [f"{fn}{extension}" for fn in filenames1]
    else:
        raise ValueError("in_path argument not recognized")
        
    if len(filenames1) == 0:
        raise IOError("No input ODD files found")
    assert all([os.path.splitext(fn)[-1]=='.mrc' and 'ODD' in fn for fn in filenames1])

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
    file_split['valid1'] = list(np.take(filenames1, rand_idx))
    file_split['valid2'] = list(np.take(filenames2, rand_idx))

    return file_split
