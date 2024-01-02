
import torch.nn as nn
import torch.nn.functional as F 

"""
use Trilinear to upsample spatially, polynomail to upsample temporally
"""
class TriLinear(nn.Module):
    """
    Parameters
    ----------
    upscale_factor : int
    """
    
    def __init__(self, upscale_factor,num_snapshots):

        super(TriLinear, self).__init__()

        self.upsacle_factor = upscale_factor
        self.num_snapshots = num_snapshots
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input Low-Resolution image as tensor

        Returns
        -------
        torch.Tensor
            Super-Resolved image as tensor

        """
        B,C,T,H,W = x.shape
        
        x = F.interpolate(x, size=(self.num_snapshots,H*self.scale_factor,W*self.scale_factor), mode='trilinear', align_corners=False)
        # bicubic upsampling
        return x