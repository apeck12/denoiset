import os
import torch
import torch.nn as nn
import denoiset.blocks as blocks


class UNet3d(nn.Module):
    """
    3D UNet architecture similar to Topaz's architecture as described 
    in Bepler, Kelley, Noble, and Berger, Nature Communications, 2020.
    """
    def __init__(self, n_filters: int=48, slope: float=0.1):
        """ Initialize a 3D U-Net. """
        super().__init__()
        
        self.encoding1 = blocks.EncodingBlock3d(1, n_filters, slope=slope)
        self.encoding2 = blocks.EncodingBlock3d(n_filters, n_filters, slope=slope)
        self.encoding3 = blocks.EncodingBlock3d(n_filters, n_filters, slope=slope)
        self.encoding4 = blocks.EncodingBlock3d(n_filters, n_filters, slope=slope)
        self.encoding5 = blocks.EncodingBlock3d(n_filters, n_filters, slope=slope)
        self.bottleneck = blocks.Bottleneck3d(n_filters, n_filters, slope=slope)
        self.decoding5 = blocks.DecodingBlock3d(2*n_filters, 2*n_filters, slope=slope)
        self.decoding4 = blocks.DecodingBlock3d(3*n_filters, 2*n_filters, slope=slope)
        self.decoding3 = blocks.DecodingBlock3d(3*n_filters, 2*n_filters, slope=slope)
        self.decoding2 = blocks.DecodingBlock3d(3*n_filters, 2*n_filters, slope=slope)
        self.decoding1 = blocks.OutBlock3d(2*n_filters+1, int(4./3*n_filters), 1, slope=slope)

        self._init_weights()

    def _init_weights(self) -> int:
        """ Initializes weights according to the Kaiming approach. """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
        
    def count_parameters(self):
        """ Count the number of model paramters. """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Pass data through encoder, bottleneck, and decoder layers 
        with skip connections.
        """
        x1 = self.encoding1(x)
        x2 = self.encoding2(x1)
        x3 = self.encoding3(x2)
        x4 = self.encoding4(x3)
        x5 = self.encoding5(x4)
        y = self.bottleneck(x5)
        y = self.decoding5(y, x4)
        y = self.decoding4(y, x3)
        y = self.decoding3(y, x2)
        y = self.decoding2(y, x1)
        y = self.decoding1(y, x)
        
        return y


def load_model_3d(filename: str) -> UNet3d:
    """
    Load saved weights for a pretrained UNet3d model.

    Parameters
    ----------
    filename: path to saved model file

    Returns
    -------
    unet_model: GPU-loaded instance of a pretrained UNet3d
    """
    unet_model = UNet3d()
    unet_model.load_state_dict(torch.load(filename))
    unet_model = unet_model.cuda()
    return unet_model


def save_model(model: nn.Module, filename: str) -> None:
    """
    Save a model's state dictionary.
    
    Parameters
    ----------
    model: nn.Module object 
    filename: filename for saving model
    """
    if os.path.dirname(filename) != "":
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)
