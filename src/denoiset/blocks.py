import torch
import torch.nn as nn
import torch.nn.functional as F


class InBlock3d(nn.Module):
    """
    Encoding block used as the first layer in the original 
    Noise2Noise paper (but not Topaz).
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        slope: float=0.1,
    ) -> None:
        """
        Parameters
        ----------
        in_channels: number of input channels
        out_channels: number of filters/output channels
        slope: negative slope used for leaky ReLU
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(slope),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(slope),
            nn.MaxPool3d(2),
        )
        
    def forward(self, x):
        return self.block(x)
    

class EncodingBlock3d(nn.Module):
    """
    3d UNet encoding block.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        slope: float=0.1,
    ) -> None:
        """
        Parameters
        ----------
        in_channels: number of input channels
        out_channels: number of filters/output channels
        slope: negative slope used for leaky ReLU
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(slope),
            nn.MaxPool3d(2),
        )
        
    def forward(self, x):
        return self.block(x)

    
class Bottleneck3d(nn.Module):
    """
    3d UNet bottleneck block.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        slope: float=0.1,
    ) -> None:
        """
        Parameters
        ----------
        in_channels: number of input channels
        out_channels: number of filters/output channels
        slope: negative slope used for leaky ReLU
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(slope),
        )
        
    def forward(self, x):
        return self.block(x)
    
    
class DecodingBlock3d(nn.Module):
    """
    3d UNet decoding block.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        slope: float=0.1,
    ) -> None:
        """
        Parameters
        ----------
        in_channels: number of input channels
        out_channels: number of filters/output channels
        slope: negative slope used for leaky ReLU
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(slope),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(slope),
        )

    def forward(self, x, skip_residual):
        """
        In addition to applying layer operations, add information 
        from the corresponding encoding layer (skip connection).
        
        To-do: test F.interpolate with
        mode='bilinear', align_corners=True
        """
        up_x = F.interpolate(x, size=tuple(skip_residual.size()[2:]), mode='nearest')
        concat_x = torch.cat([up_x, skip_residual], dim=1)
        return self.block(concat_x)
    
    
class OutBlock3d(nn.Module):
    """
    3d UNet final decoding block.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        slope: float=0.1,
    ) -> None:
        """
        Parameters
        ----------
        in_channels: number of input channels
        mid_channels: intermediate number of channels
        out_channels: number of filters/output channels
        slope: negative slope used for leaky ReLU
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            nn.LeakyReLU(slope),
            nn.Conv3d(mid_channels, int(mid_channels/2), 3, padding=1),
            nn.LeakyReLU(slope),
            nn.Conv3d(int(mid_channels/2), out_channels, 3, padding=1),
        )

    def forward(self, x, skip_residual):
        """
        In addition to applying layer operations, add information 
        from the corresponding encoding layer (skip connection).
        
        To-do: test F.interpolate with
        mode='bilinear', align_corners=True
        """
        up_x = F.interpolate(x, size=tuple(skip_residual.size()[2:]), mode='nearest')
        concat_x = torch.cat([up_x, skip_residual], dim=1)
        return self.block(concat_x)
