import numpy as np


def atleast_nd(
    arr: np.ndarray,
    ndim: int,
):
    """
    Expand input array to be n-dimensional, equivalent to
    arr[:,np.newaxis,np.newaxis,...].
    
    Parameters
    ----------
    arr: array to expand
    ndim: target dimensionality
    
    Returns
    -------
    arr expanded to target dimensionality
    """
    new_shape = arr.shape + (1,) * (ndim - arr.ndim)
    return arr.reshape(new_shape)


def normalize(
    arr: np.ndarray,
    along_first_axis=True,
    return_stats=False,
) -> np.ndarray | tuple:
    """
    Z-score normalization, potentially per patch or subvolume
    rather than on the full input array.
    
    Parameters
    ----------
    arr: array to normalize
    along_first_axis: normalize per patch or subvolume
    return_stats: also return mean and sigma
    
    Returns
    -------
    normalized: normalized array
    mu: mean(s), optional
    sigma: standard deviation(s), optional
    """
    if along_first_axis:
        mu = np.mean(arr, axis=tuple(np.arange(1,arr.ndim)))
        sigma = np.std(arr, axis=tuple(np.arange(1,arr.ndim)))
        normalized = arr - atleast_nd(mu, arr.ndim) 
        normalized /= atleast_nd(sigma, arr.ndim)
    
    else:
        mu = np.mean(arr)
        sigma = np.std(arr)
        normalized = (arr - mu) / sigma
        
    if not return_stats:
        return normalized
    else:
        return normalized, mu, sigma


def augment(
    data1: np.ndarray,
    data2: np.ndarray,
) -> tuple[np.ndarray,np.ndarray]:
    """
    Randomly rotate the paired input data, flip axes, and swap.
    Input data must either be square patches or cubic subvolumes.

    Parameters
    ----------
    data1: np.array, patch or subvolume
    data2: np.array, paired patch or subvolume

    Returns
    -------
    list of augmented data1, data2
    """
    # check dimensions
    if data1.ndim == 2:
        ax_list = [(0,1)]
    elif data1.ndim == 3:
        ax_list = [(0,1), (0,2), (1,2)]
    else:
        raise ValueError("Invalid data dimensions, must be 2d or 3d")

    # randomly flip 
    for ax in range(data1.ndim):
        if np.random.rand() > 0.5:
            data1 = np.flip(data1, axis=ax)
            data2 = np.flip(data2, axis=ax)

    # randomly rotate
    for ax in ax_list:
        rot_k = np.random.randint(0, high=4)
        data1 = np.rot90(data1, k=rot_k, axes=ax)
        data2 = np.rot90(data2, k=rot_k, axes=ax)

    # randomly swap
    if np.random.rand() > 0.5:
        return np.ascontiguousarray(data1), np.ascontiguousarray(data2)
    else:
        return np.ascontiguousarray(data2), np.ascontiguousarray(data1)
