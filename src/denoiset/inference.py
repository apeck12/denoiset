import numpy as np
import itertools
import os
import time
import torch
import denoiset.dataio as dataio
from denoiset.model import UNet3d
from denoiset.model import load_model_3d


def get_bounds_one_axis(dim: int, length:int, padding:int) -> list:
    """
    Determine the boundaries to achieve the desired crop
    length with overlap.

    Parameters
    ----------
    dim: max number of pixels along axis
    length: subvolume side length before padding
    padding: number of pixels for padding/overlap

    Returns
    -------
    pts: list of tuples of (start,end) pixels to crop
    """
    pts = [0]
    while pts[-1] < dim:
        if len(pts)%2!=0:
            pts.append(pts[-1]+length+padding)
        else:
            pts.append(pts[-1]-padding)
    if pts[-1] > dim:
        pts[-1] = dim
    pts = [(pts[i], pts[i+1]) for i in np.arange(len(pts))[::2]]
    return pts


def get_bounds_3d(shape: tuple, length: int, padding: int) -> tuple:
    """
    Get boundaries for extracting padded subvolumes from a volume, 
    removing the padding (after denoising for example), and then 
    tiling a new volume with the processed subvolumes. Output arrays
    have shape [number of subvolumes, 6], where each entry corresponds
    to a [xstart,xend,ystart,yend,zstart,zend] subvolume coordinates.
    
    Parameters
    ----------
    shape: volume shape
    length: subvolume side length before padding
    padding: number of pixels for padding/overlap
    
    Returns
    -------
    ibounds: initial cropping bounds for cropping padded subvolumes
    rbounds: filling bounds for where to place the cropped subvolume
    sbounds: subvolume cropping bounds for eliminating padded region
    """
    # enforce even padding
    if padding % 2 != 0:
        padding -= 1
    hpadding = int(padding/2)
    
    # determine bounds for extracting padded subvolumes
    xpts = get_bounds_one_axis(shape[0], length, padding)
    ypts = get_bounds_one_axis(shape[1], length, padding)
    zpts = get_bounds_one_axis(shape[2], length, padding)
    
    ibounds = np.array(list(itertools.product(xpts, ypts, zpts)))
    ibounds = ibounds.reshape(ibounds.shape[0],6)

    # determine bounds for stitching subvolumes into volume
    hpadding = int(padding/2)
    rbounds = ibounds.copy()
    rbounds[:,::2] += hpadding
    rbounds[:,1::2] -= hpadding
    rbounds[ibounds==0] = 0
    rbounds[:,:2][ibounds[:,:2]==shape[0]] = shape[0]
    rbounds[:,2:4][ibounds[:,2:4]==shape[1]] = shape[1]
    rbounds[:,4:][ibounds[:,4:]==shape[2]] = shape[2]

    # determine bounds for cropping subvolumes, accounting for padding
    sbounds = rbounds - ibounds
    sbounds[:,1][sbounds[:,1]==0] = rbounds[sbounds[:,1]==0][:,1]-rbounds[sbounds[:,1]==0][:,0] + sbounds[sbounds[:,1]==0][:,0]
    sbounds[:,3][sbounds[:,3]==0] = rbounds[sbounds[:,3]==0][:,3]-rbounds[sbounds[:,3]==0][:,2] + sbounds[sbounds[:,3]==0][:,2]
    sbounds[:,5][sbounds[:,5]==0] = rbounds[sbounds[:,5]==0][:,5]-rbounds[sbounds[:,5]==0][:,4] + sbounds[sbounds[:,5]==0][:,4]

    return ibounds, rbounds, sbounds


def denoise_volume(
    volume: np.ndarray, 
    model: UNet3d, 
    length: int, 
    padding: int, 
) -> np.ndarray:
    """
    Denoise one volume by applying a pre-trained model
    to overlapping patches. Variant in which the full
    volume rather than subvolumes are normalized.
    
    Parameters
    ----------
    volume: tomogram to denoise
    model: pretrained model
    length: subvolume side length before padding
    padding: number of pixels for padding/overlap
    
    Returns
    -------
    volume_d: denoised volume
    """
    if not next(model.parameters()).is_cuda:
        model = model.cuda()
    
    ibounds, rbounds, sbounds = get_bounds_3d(volume.shape, length, padding)
    volume = torch.from_numpy(volume)
    volume = volume.cuda()
    volume_d = torch.zeros_like(volume)
    volume = volume.unsqueeze(0).unsqueeze(0)
    mu, sigma = volume.mean(), volume.std()
    volume = (volume - mu) / sigma

    for i in range(ibounds.shape[0]):
        volume_i = volume[:,:,ibounds[i][0]:ibounds[i][1],ibounds[i][2]:ibounds[i][3],ibounds[i][4]:ibounds[i][5]]
        with torch.no_grad():
            volume_i = model(volume_i).squeeze()
        volume_i = volume_i[sbounds[i][0]:sbounds[i][1],sbounds[i][2]:sbounds[i][3],sbounds[i][4]:sbounds[i][5]]
        volume_d[rbounds[i][0]:rbounds[i][1],rbounds[i][2]:rbounds[i][3],rbounds[i][4]:rbounds[i][5]] = volume_i
        
    volume_d = sigma * volume_d + mu
    return volume_d.cpu().numpy()


class Denoiser3d:
    
    def __init__(
        self,
        fn_model: str,
        out_dir: str,
        length: int,
        padding: int,
    ) -> None:
        """
        Set up class to denoise volumes.
        
        Parameters
        ----------
        fn_model: path to pretrained model
        out_dir: output directory
        length: subvolume side length before padding
        padding: number of pixels for padding/overlap
        """
        self.model = load_model_3d(fn_model)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.length = length
        self.padding = padding
        self.processed = []
        self.apix = None
        
    def denoise(
        self,
        fnames: list,
    ) -> None:
        """
        Denoise the listed files.
        
        Parameters
        ----------
        fnames: files (full path) to process
        """
        for i,fn in enumerate(fnames):
            if self.apix == None:
                self.apix = dataio.get_voxel_size(fn)
            
            volume = dataio.load_mrc(fn).copy()
            volume = denoise_volume(
                volume, self.model, self.length, self.padding,
            )
            
            fn_out = os.path.join(self.out_dir, fn.split("/")[-1])
            dataio.save_mrc(
                volume.astype(np.float32), fn_out, self.apix,
            )
            self.processed.append(fn)
        
    def process(
        self,
        in_dir: str,
        pattern: str="*Vol.mrc",
        exclude_tags: list=["ODD","EVN"],
        filenames: str=None,
        t_interval: float=300,
        t_exit: float=1800,
    ) -> None:
        """ 
        Denoise available tomograms, potentially in live mode.
        
        Parameters
        ----------
        in_dir: input directory
        pattern: glob-expandable pattern contained by file basenames
        exclude_tags: substring(s) for file exclusion
        t_interval: seconds to wait before checking for new files
        t_exit: seconds to wait after last finding new files before exiting
        """
        start_time = time.time()
        while True:
            if filenames is None:
                fnames = dataio.expand_filelist(
                    in_dir, pattern, exclude_tags,
                )
                fnames = [fn for fn in fnames if fn not in self.processed]
            else:
                fnames = np.loadtxt(filenames, dtype=str, ndmin=1)
                fnames = [os.path.join(in_dir, f"{fn}_Vol.mrc") for fn in fnames]
            print(time.strftime("%X %x %Z"), f": Found {len(fnames)} new files to process")
            
            if len(fnames) > 0:
                self.denoise(fnames)
                start_time = time.time()
                
            time.sleep(t_interval)
            t_elapsed = time.time() - start_time
            if t_elapsed > t_exit:
                break
