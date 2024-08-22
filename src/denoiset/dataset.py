import numpy as np

def get_random_coords_3d(
    shape: tuple[int,int,int],
    length: int,
    n_extract: int,
    rng: np.random._generator.Generator=None,
    f_omit: float=0.03,
)-> np.ndarray:
    """
    Get random 3d coordinates within a volume space.

    Parameters
    ----------
    shape: volume shape
    length: subvolume side length in voxels
    n_extract: number of subvolumes to extract
    rng: numpy random generator object
    f_omit: fraction of volume dimensions to omit along each edge

    Returns
    -------
    (x,y,z) extraction coordinates corner of shape (n_extract, 3)
    """
    buffer = np.around(np.array(shape) * f_omit).astype(int)
    if rng is None:
        rng = np.random.default_rng()
        
    xc = rng.choice(range(buffer[0], shape[0]-length-buffer[0]), size=n_extract)
    yc = rng.choice(range(buffer[0], shape[1]-length-buffer[1]), size=n_extract)
    zc = rng.choice(range(buffer[0], shape[2]-length-buffer[2]), size=n_extract)
    
    return np.array([xc,yc,zc]).T


def extract_subvolumes(
    volume: np.ndarray,
    coords: np.ndarray,
    length: int=96,
) -> np.array:
    """
    Extract subvolumes based on the specified coordinates.
    
    Parameters
    ----------
    volume: volume to extract from
    coords: coordinates of each subvolume's corner
    length: side length of cubic subvolume
    
    Returns
    -------
    array of subvolumes of shape (n_extract, length, length, length)
    """
    zcoords = zip(coords[:,0],coords[:,1],coords[:,2])
    return np.array([volume[x1:x1+length,y1:y1+length,z1:z1+length] for x1,y1,z1 in zcoords])

