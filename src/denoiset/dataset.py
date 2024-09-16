from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

import denoiset.dataio as dataio
import denoiset.transform as transform

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


class PairedData(Dataset):
    """
    Abstract class to synchronize the handling of 2D and 3D data.
    
    Attributes
    ----------
    filenames1: list of pair 1 filenames 
    filenames2: list of pair 2 filenames
    length: edge length in pixels
    n_extract: number of data pairs to extract
    rng: random generator object if seed is fixed
    f_omit: fraction of micrograph/volume border to omit
    n_load: number of micrographs or tomograms to load at once
    """        
    def __init__(
        self, 
        filenames1: list, 
        filenames2: list, 
        length: int, 
        n_extract: int, 
        rng: np.random._generator.Generator=None,
        f_omit: float=0.03,
    ) -> None:
        self.filenames1 = filenames1
        self.filenames2 = filenames2
        self.length = length
        self.n_extract = n_extract
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        self.f_omit = f_omit
        self.pairs1, self.pairs2 = None, None
        self.get_num_load()
    
    def randomize_filenames(self) -> None:
        """ Randomize the order of paired filenames. """
        indices = self.rng.permutation(len(self.filenames1))
        self.filenames1 = [self.filenames1[i] for i in indices]
        self.filenames2 = [self.filenames2[i] for i in indices]
    
    def randomize_data_pairs(self) -> None:
        """ Randomize the order of the data pairs. """
        indices = self.rng.permutation(len(self.pairs1))
        self.pairs1 = self.pairs1[indices]
        self.pairs2 = self.pairs2[indices]
        
    def get_num_load(self) -> None:
        pass
    
    def __len__(self):
        """ Return the number of data pairs. """
        return self.n_extract * len(self.filenames1)
    
    def __getitem__(self, idx):
        pass
    
    
class PairedTomograms(PairedData):
    """
    Child class of PairedData for tomogram inputs.
    """
    def get_num_load(self):
        """
        Determine the number of tomograms to store in memory.
        """
        threshold = 5250000000 # approx number of voxels
        self.n_load = int(threshold / (self.n_extract * self.length**3))
        self.n_load = min(self.n_load, len(self.filenames1))
        
    def get_paired_subvolumes(
        self, 
        volume1: np.ndarray, 
        volume2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve paired subvolumes.
        
        Parameters
        ----------
        volume1: first member of volume pair
        volume2: second member of volume pair
        
        Returns
        -------
        subvolumes1: array of subvolumes
        subvolumes2: array of paired subvolumes
        """
        assert volume1.shape == volume2.shape
        coords = get_random_coords_3d(
            volume1.shape,
            self.length,
            self.n_extract,
            rng = self.rng,
            f_omit = self.f_omit,
        )
        subvolumes1 = extract_subvolumes(volume1, coords, self.length)
        subvolumes2 = extract_subvolumes(volume2, coords, self.length)
        return subvolumes1, subvolumes2
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve a subvolume pair. Randomization is achieved by
        shuffling the order of files between training epochs in
        addition to randomly drawing extraction coordinates.
        
        Due to memory constraints, a limited number of subvolumes
        is loaded at once. The index_fn variable is used to track
        which index to load relative to the subvolumes that are in
        already memory.
        
        Parameters
        ----------
        index: index of the retrieved subvolume
        
        Returns
        -------
        subvolume1: augmented subvolume
        subvolume2: augmented subvolume pair
        """
        if index == 0:
            self.randomize_filenames()
            
        index_fn = index // self.n_extract 
        index_pair = index % (self.n_extract * self.n_load) 
        
        # draw new subvolumes if none left from current volume
        if index_pair == 0:
            self.pairs1, self.pairs2 = None, None
            for n in tqdm(range(self.n_load), desc="Loading tomograms"):
                if index_fn + n < len(self.filenames1):
                    volume1 = dataio.load_mrc(self.filenames1[index_fn+n])
                    volume2 = dataio.load_mrc(self.filenames2[index_fn+n])
                    subvolumes1, subvolumes2 = self.get_paired_subvolumes(
                        volume1,
                        volume2,
                    )
                    if self.pairs1 is None:
                        self.pairs1 = subvolumes1
                        self.pairs2 = subvolumes2
                    else:
                        self.pairs1 = np.concatenate((self.pairs1, subvolumes1))
                        self.pairs2 = np.concatenate((self.pairs2, subvolumes2))
                        
            self.pairs1 = transform.normalize(self.pairs1, along_first_axis=True)
            self.pairs2 = transform.normalize(self.pairs2, along_first_axis=True)
            self.randomize_data_pairs()
            
        subvolume1, subvolume2 = transform.augment(self.pairs1[index_pair], self.pairs2[index_pair])
        return np.expand_dims(subvolume1, axis=0), np.expand_dims(subvolume2, axis=0)
