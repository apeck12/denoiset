import numpy as np
import denoiset.dataset as dataset


def test_extract_3d():
    """
    Verify that returns of get_random_coords_3d and extract_subvolumes 
    functions have the right shape and that omission of border regions
    is working correctly.
    """
    a = np.arange(20*10*30).reshape(20,10,30)
    a[1:-1,2:-2,2:-2] = 0
    n_extract = 100
    length = 3

    coords = dataset.get_random_coords_3d(a.shape, length, n_extract, f_omit=0.1)
    assert coords.shape == (n_extract, 3)

    subvolumes = dataset.extract_subvolumes(a, coords, length)
    assert subvolumes.shape == (n_extract, length, length, length)
    assert np.count_nonzero(subvolumes) == 0

    coords = dataset.get_random_coords_3d(a.shape, length, n_extract, f_omit=0)
    subvolumes = dataset.extract_subvolumes(a, coords, length)
    assert np.count_nonzero(subvolumes) != 0
