import numpy as np
import pytest
import shutil
import os

import denoiset.dataio as dataio


def test_mrc_functions():
    """
    Test mrc functions: loading, retrieving voxel size and saving.
    """
    apix = 5
    out_name = "test.mrc"

    vol = np.random.randn(5, 3).astype(np.float32)
    dataio.save_mrc(vol, out_name, apix=apix)
    assert dataio.get_voxel_size(out_name) == apix

    same_vol = dataio.load_mrc(out_name)
    np.testing.assert_array_equal(vol, same_vol)
    os.remove(out_name)

    
class TestFileSplits:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        cwd = os.path.dirname(os.path.realpath(__file__))
        self.fnames = np.loadtxt(os.path.join(cwd, "input_names.txt"), dtype=str)
        self.test_dir = "test_dataio"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        self.make_mock_volumes()
        
    def make_mock_volumes(self):
        """ Make mock volumes for testing purposes. """
        for i,fn in enumerate(self.fnames):
            data = np.random.randn(3,4,4).astype(np.float32)
            dataio.save_mrc(data, os.path.join(self.test_dir, fn))
            
    def test_expand_filelist(self):
        """ Check for expected number of files and filenames. """
        n_expected = len(self.fnames) / 3
        for substring in ['EVN', 'ODD']:
            filelist = dataio.expand_filelist(self.test_dir, f"*{substring}_Vol.mrc")
            filelist = set([os.path.basename(fn) for fn in filelist])
            reflist = set([fn for fn in self.fnames if substring in fn])
            assert filelist == reflist
            assert len(filelist) == n_expected

        filelist = dataio.expand_filelist(self.test_dir, f"*Vol.mrc", exclude_tags=['EVN', 'ODD'])
        filelist = set([os.path.basename(fn) for fn in filelist])
        reflist = set([fn for fn in self.fnames if 'ODD' not in fn and 'EVN' not in fn])
        assert filelist == reflist
        assert len(filelist) == n_expected
        
    def test_get_split_filenames(self):
        """ Various tests of the get_split_filenames function. """
        # verify expected number of files for validation fraction
        f_val = np.random.uniform(low=0, high=0.4)
        file_split = dataio.get_split_filenames(self.test_dir, f_val, pattern="*ODD_Vol.mrc")
        assert len(file_split['test1']) == len(file_split['test2']) == int(np.around(f_val * 10))

        # verify no overlap between train and test; train1/train2 and test1/test2 match
        assert [fn.replace('ODD', 'EVN') for fn in file_split['test1']] == file_split['test2']
        assert [fn.replace('ODD', 'EVN') for fn in file_split['train1']] == file_split['train2']
        assert all([fn not in file_split['test1'] for fn in file_split['train2']])
        assert all([fn not in file_split['test2'] for fn in file_split['train2']])

        # check that random seed works as expected
        random_seed = np.random.randint(0, high=5)
        rng = np.random.default_rng(random_seed)
        file_split1 = dataio.get_split_filenames(self.test_dir, f_val, pattern="*ODD_Vol.mrc", rng=rng)
        rng = np.random.default_rng(random_seed)
        file_split2 = dataio.get_split_filenames(self.test_dir, f_val, pattern="*ODD_Vol.mrc", rng=rng)
        assert file_split1['train1'] == file_split2['train1']
        assert file_split1['test1'] == file_split2['test1']
        random_seed = np.random.randint(6, high=10)
        rng = np.random.default_rng(random_seed)
        file_split2 = dataio.get_split_filenames(self.test_dir, f_val, pattern="*ODD_Vol.mrc", rng=rng)
        assert file_split1['train1'] != file_split
        
        shutil.rmtree(self.test_dir)
