import numpy as np
import denoiset.transform as transform


def test_normalize():
    """
    Test array normalization for both 2d and 3d.
    """

    patches = np.random.rand(20,4,4) * 5 + 2
    subvolumes = np.random.rand(20,4,4,4) * 5 + 2
    
    for a in [patches, subvolumes]:

        norm_a, mu, sigma = transform.normalize(a, return_stats=True)
        ref = np.array([(a[i] - np.mean(a[i])) / np.std(a[i]) for i in range(a.shape[0])])
        assert np.allclose(norm_a, ref)
        assert np.allclose(mu, np.array([np.mean(a[i]) for i in range(a.shape[0])]))
        assert np.allclose(sigma, np.array([np.std(a[i]) for i in range(a.shape[0])]))
    
        norm_a, mu, sigma = transform.normalize(a, along_first_axis=False, return_stats=True)
        ref = (a - np.mean(a)) / np.std(a)
        assert np.allclose(norm_a, ref)
        assert mu == np.mean(a)
        assert sigma == np.std(a)

    
def test_augment():
    """ Test that paired data are correspondingly transformed. """
    for dims in [(3,9),(3,3,3)]:
        x = np.arange(27).reshape(dims)
        d1, d2 = transform.augment(x, x+1)
        assert np.allclose(1, np.abs(d1 - d2))
