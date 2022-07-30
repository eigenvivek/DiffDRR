import torch

from .projectors.siddon import Siddon
from .utils.backend import get_device
from .utils.camera import Detector


class DRR:
    def __init__(
        self,
        volume,
        spacing,
        height,
        delx,
        width=None,
        dely=None,
        device="cpu",
        return_grads=False
    ):
        """
        Class for generating DRRs.

        Inputs
        ------
        volume : np.ndarray
            CT volume.
        spacing : tuple of float
            The spacing of the volume.
        height : int
            The height of the DRR.
        width : int, optional
            The width of the DRR. If not provided, it is set to `height`.
        delx : float
            The x-axis pixel size.
        dely : float, optional
            The y-axis pixel size. If not provided, it is set to `delx`.
        device : str
            Compute device, either "cpu", "cuda", or "mps".
        return_grads : bool, optional
            If True, return differentiable vectors for rotations and translations
            
        Returns
        -------
        self : DRR
            
        """
        self.device = get_device(device)
        self.return_grads = return_grads
        
        # Initialize the X-ray detector
        width = height if width is None else width
        dely = delx if dely is None else delx
        self.detector = Detector(height, width, delx, dely, device)
        
        # Initialize the Projector
        self.siddon = Siddon(volume, spacing, device)
    
    def project(self, sdr, theta, phi, gamma, bx, by, bz):
        """
        Generate a DRR from given spatial parameters.
        
        Inputs
        ------
            sdr   : Source-to-Detector radius (half of the source-to-detector distance)
            theta : Azimuthal angle
            phi   : Polar angle
            gamma : Plane rotation angle
            bx    : X-dir translation
            by    : Y-dir translation
            bz    : Z-dir translation
        """
        sdr = torch.tensor([sdr], dtype=torch.float32, device=self.device)
        rotations = torch.tensor([theta, phi, gamma], dtype=torch.float32, device=self.device, requires_grad=True)
        translations = torch.tensor([bx, by, bz], dtype=torch.float32, device=self.device, requires_grad=True)
        source, rays = self.detector.make_xrays(sdr, rotations, translations)
        drr = self.siddon.raytrace(source, rays)
        if self.return_grads is False:
            return drr
        else:
            return drr, rotations, translations
