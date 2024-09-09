import numpy as np
import pytest
import os
import shutil
import denoiset.dataset as dataset
import denoiset.dataio as dataio

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


class TestPairedTomograms:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        cwd = os.path.dirname(os.path.realpath(__file__))
        self.fnames = np.loadtxt(os.path.join(cwd, "input_names.txt"), dtype=str)
        self.test_dir = "test_datio"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        self.make_mock_volumes()
        self.make_pd_class()
        
    def make_mock_volumes(self):
        """ Make mock volumes for testing purposes. """
        for i,fn in enumerate(self.fnames):
            data = np.random.randn(10,15,15).astype(np.float32)
            dataio.save_mrc(data, os.path.join(self.test_dir, fn))
 
    def make_pd_class(self):
        """ Make a PairedTomogram class. """
        length = 4
        n_extract = 5
        d_fn = dataio.get_split_filenames(self.test_dir, 0.1, pattern="*ODD_Vol.mrc")
        self.paired_data = dataset.PairedTomograms(
            d_fn['train1'], 
            d_fn['train2'], 
            length, 
            n_extract,
        )
        self.paired_data.__getitem__(0)
        
    def test_get_paired_subvolumes(self):
        """ Test that corresponding subvolumes are extracted. """
        a = np.random.randn(10,10,8)
        sv1, sv2 = self.paired_data.get_paired_subvolumes(a, a)
        assert np.array_equal(sv1, sv2)
        
    def test_randomize_filenames(self):
        """ Test randomization of filenames. """
        order1 = self.paired_data.filenames1
        self.paired_data.randomize_filenames()
        assert order1 != self.paired_data.filenames1
        assert [fn.replace('ODD', 'EVN') for fn in self.paired_data.filenames1] == self.paired_data.filenames2
        
    def test_randomize_data_pairs(self):
        """ Test randomization of data pair order. """
        order1 = self.paired_data.pairs1[3]
        self.paired_data.randomize_data_pairs()
        assert not np.array_equal(order1, self.paired_data.pairs1[3])
        shutil.rmtree(self.test_dir)
