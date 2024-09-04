import random
import numpy as np
import denoiset.inference as inference


def patchify_3d(length, padding, shape):
    """
    For a given subvolume size (length) and padding and volume 
    shape, check bounds correctness by cropping subvolumes and
    patching them into a new volume. Correctness is verified by
    verifying that the original and patched volumes match.
    """
    ibounds, rbounds, sbounds = inference.get_bounds_3d(shape, length, padding)
    volume = np.random.randn(shape[0],shape[1],shape[2])
    d_volume = np.zeros(volume.shape)
    for i in range(ibounds.shape[0]):
        subvolume = volume[ibounds[i][0]:ibounds[i][1],ibounds[i][2]:ibounds[i][3],ibounds[i][4]:ibounds[i][5]]
        cropvolume = subvolume[sbounds[i][0]:sbounds[i][1],sbounds[i][2]:sbounds[i][3],sbounds[i][4]:sbounds[i][5]]
        d_volume[rbounds[i][0]:rbounds[i][1],rbounds[i][2]:rbounds[i][3],rbounds[i][4]:rbounds[i][5]] = cropvolume
    
    assert np.sum(np.abs(volume - d_volume)) == 0


def test_get_bounds_3d():
    """
    Validate bounds generation for patching and stitching in 3d.
    """
    length = random.choice([64,80,96,124])
    padding = random.choice([24,32])
    shape = (length,length-4,length-5)
    patchify_3d(length, padding, shape)
    
    shape = random.sample(range(80,480),3)
    patchify_3d(length, padding, shape)
    
