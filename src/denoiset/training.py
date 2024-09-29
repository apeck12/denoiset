import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import denoiset.dataio as dataio
import denoiset.dataset as dataset
import denoiset.tracker as tracker
import denoiset.inference as inference
from denoiset.model import load_model_3d, generate_model_3d, save_model

class Trainer3d:
    
    def __init__(
        self,
        in_path: str,
        out_path: str,
        fn_model: str=None,
        seed: int=None,
        optimizer: str="adagrad",
        learning_rate: float=0.001,
        batch_size: int=8,
        val_fraction: float=0.1,
        pattern: str="*ODD_Vol.mrc",
        extension: str="_ODD_Vol.mrc",
        length: int=96,
        n_extract: int=200,
    ) -> None:
        """
        Set up class for Noise2Noise training on tomography data. 
        
        Parameters
        ----------
        in_path: directory or text file of training tomograms
        out_path: output directory
        fn_model: path to a pretrained model
        seed: fixed random seed value
        optimizer: optimizer type
        learning_rate: optimizer learning rate
        batch_size: number of paired subvolumes per batch  
        val_fraction: fraction of tomograms used for validation
        pattern: glob-expandable pattern for ODD tomograms
        extension: suffix for ODD tomograms
        length: side length for subvolume extraction
        n_extract: number of subvolumes to extract per tomogram
        """
        self.rng = np.random.default_rng(seed)
        self.set_model(fn_model, seed)
        self.loss_fn = nn.MSELoss()
        self.set_optimizer(optimizer, learning_rate)
        self.batch_size = batch_size
        self.set_dataloaders(
            in_path, 
            val_fraction, 
            pattern, 
            extension, 
            length,
            n_extract,
        )
        self.out_path = out_path
        
    def set_model(
        self,
        fn_model: str=None,
        seed: int=None,
    ) -> None:
        """
        Set up UNet3d model, optionally from a pretrained model
        or with weights initialized using a fixed random seed
        if fn_model or seed_value is supplied, respectively.
        """
        if fn_model is not None:
            self.model = load_model_3d(fn_model)
            self.pretrained = True
        else:
            self.model = generate_model_3d(seed)
            self.pretrained = False
        
    def set_optimizer(
        self,
        optimizer: str,
        learning_rate: float,
    ) -> None:
        """
        Set up optimizer. 
        """
        if optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(
                self.model.parameters(), lr=learning_rate,
            )
        elif optimizer == "adamw":
            self.optimizer =  torch.optim.AdamW(
                self.model.parameters(), lr=learning_rate,
            )
        else:
            raise NotImplementedError
    
    def set_dataloaders(
        self,
        in_path: str,
        val_fraction: float,
        pattern: str, 
        extension: str, 
        length: int=96,
        n_extract: int=100,
    ) -> None:
        """
        Generate Dataloaders for training and validation sets.
        """
        file_split = dataio.get_split_filenames(
            in_path,
            val_fraction,
            pattern=pattern,
            extension=extension,
            exclude_tags=[],
            rng=self.rng,
            length=length,
        )

        dataset_train = dataset.PairedTomograms(
            file_split['train1'], file_split['train2'], length, n_extract,
        )
        dataset_valid = dataset.PairedTomograms(
            file_split['valid1'], file_split['valid2'], length, n_extract,
        )
        self.dataloader_train = DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=False, num_workers=0,
        )
        self.dataloader_valid = DataLoader(
            dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=0,
        )

    def set_denoising_volumes(self, n_denoise: int=0):
        """ Select volumes to optionally denoise each epoch. """
        if n_denoise == 0:
            self.repr_volumes = np.empty(0)
        else:
            filenames = np.concatenate((
                self.dataloader_train.dataset.filenames1,
                self.dataloader_valid.dataset.filenames2,
            ))
            filenames = self.rng.choice(
                filenames, n_denoise, replace=False,
            )
            filenames = [fn.replace('ODD_', '') for fn in filenames]
            if all([os.path.exists(fn) for fn in filenames]):
                self.repr_volumes = filenames
                self.apix = dataio.get_voxel_size(self.repr_volumes[0])
            else:
                print("Warning! Full volumes not at expected path")
                self.repr_volumes = np.empty(0)

    def denoise_repr_volumes(self, epoch: int, dlength: int, dpadding: int):
        """ Denoise representative volumes for visual inspection. """
        for vol_path in tqdm(self.repr_volumes, desc="Denoising representative tomograms"):
            volume = dataio.load_mrc(vol_path).copy()
            volume = inference.denoise_volume(volume, self.model, dlength, dpadding)
            basename = f"{os.path.splitext(os.path.basename(vol_path))[0]}_epoch{epoch}.mrc"
            dataio.save_mrc(
                volume, os.path.join(self.out_path, basename), self.apix,
            )
                
    def evaluate(self, epoch):
        """ Evaluate model on validation data. """

        tr_loss = tracker.AverageMeter()
        with torch.no_grad():
            for i,(source,target) in enumerate(self.dataloader_valid):
                source = source.cuda()
                target = target.cuda()
                source_out = self.model(source)
                loss = self.loss_fn(source_out, target)
                tr_loss.update(loss.item(), target.size(0))

                print(f'Epoch: [{epoch}][{i+1}/{len(self.dataloader_valid)}]\t'
                      f'Loss {tr_loss.val:.4f} ({tr_loss.avg:.4f})')

        return tr_loss.avg

    def train_epoch(self, epoch):
        """ 
        Train model for one epoch, tracking the loss.
        """
        tr_loss = tracker.AverageMeter()
        tr_std = tracker.AverageMeter()
        
        for i,(source,target) in enumerate(self.dataloader_train):
            # load subvolume pair onto device
            source = source.cuda()
            target = target.cuda()

            # apply model to one image, calculate loss w.r.t. its pair
            source_out = self.model(source)
            loss = self.loss_fn(source_out, target)
            tr_loss.update(loss.item(), target.size(0))
            tr_std.update(float(torch.mean(torch.std(source_out, dim=(1,2,3,4)))), target.size(0))

            # perform a backward pass, update weights, zero gradients
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # gather stats and report
            print(f'Epoch: [{epoch}][{i+1}/{len(self.dataloader_train)}]\t'
                  f'Loss {tr_loss.val:.4f} ({tr_loss.avg:.4f})\t'
                  f'Std dev {tr_std.val:.4f} ({tr_std.avg:.4f})')

        return tr_loss.avg, tr_std.avg
            
    def train(self, n_epochs: int=20, save_epoch: bool=True, n_denoise: int=0, dlength: int=128, dpadding: int=24):
        """ 
        Train model, evaluating on validation data after each epoch. 
        """
        os.makedirs(self.out_path, exist_ok=True)
        logger = tracker.Logger(
            os.path.join(self.out_path, "training_stats.csv"),
            columns=['epoch', 'loss_train', 'loss_valid', 'std_dev'],
        )
        self.set_denoising_volumes(n_denoise)

        for epoch in range(n_epochs):
            if self.pretrained and epoch==0:
                valid_loss = self.evaluate(epoch)
                if len(self.repr_volumes) > 0:
                    self.denoise_repr_volumes(epoch, dlength, dpadding)
                logger.add_entry([epoch, 0, np.around(valid_loss,4), 0], write=True)
            
            print('EPOCH {}:'.format(epoch+1))
            self.model.train(True)
            train_loss, std_dev = self.train_epoch(epoch+1)
            self.model.eval()
            valid_loss = self.evaluate(epoch+1)

            logger.add_entry([epoch+1, np.around(train_loss,4), np.around(valid_loss,4), np.around(std_dev, 4)], write=True)
            if save_epoch:
                save_model(self.model, os.path.join(self.out_path, f"epoch{epoch+1}.pth"))

            if len(self.repr_volumes) > 0:
                self.denoise_repr_volumes(epoch+1, dlength, dpadding)
                    
