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
        """
        self.device = get_device(device)
        
        # Initialize the X-ray detector
        width = height if width is None else width
        dely = delx if dely is None else delx
        self.detector = Detector(height, width, delx, dely, device)
        
        # Initialize the Projector
        self.siddon = Siddon(volume, spacing, device)
    
    def project(self, sdr, theta, phi, gamma, bx, by, bz, return_grads=False):
        """
        Generate a DRR from given spatial parameters.
        
        Inputs
        ------
        Projector parameters:
            sdr   : Source-to-Detector radius (half of the source-to-detector distance)
            theta : Azimuthal angle
            phi   : Polar angle
            gamma : Plane rotation angle
            bx    : X-dir translation
            by    : Y-dir translation
            bz    : Z-dir translation
        return_grads : bool, optional
            If True, return differentiable vectors for rotations and translations
        """
        sdr = torch.tensor([sdr], dtype=torch.float32, device=self.device)
        rotations = torch.tensor([theta, phi, gamma], dtype=torch.float32, device=self.device, requires_grad=True)
        translations = torch.tensor([bx, by, bz], dtype=torch.float32, device=self.device, requires_grad=True)
        drr = self._project(sdr, rotations, translations)
        if return_grads is False:
            return drr
        else:
            return drr, sdr, rotations, translations
        
    def _project(self, sdr, rotations, translations):
        """Helper function to do projection with preformed tensors."""
        source, rays = self.detector.make_xrays(sdr, rotations, translations)
        drr = self.siddon.raytrace(source, rays)
        return drr
        
    def optimize(
        self,
        true_drr,
        loss_fn,
        alpha1=7.5e1,
        alpha2=5.3e-2,
        n_iters=500,
        earlystop=None,
        verbose=False,
        **init_detector
    ):
        """
        Gradient-based optimization loop with repeated DRR synthesis.
        
        Inputs
        ------
        true_drr : torch.Tensor
            Ground truth DRR that we are trying to converge to.
        loss_fn : function
            Function to compute loss between two DRRs.
        alpha1 : float
            Update rate for rotational parameters.
        alpha2 : float
            Update rate for translational parameters.
        n_iters : int
            Number of gradient descent steps to take.
        earlystop : float
            Cutoff for early stop criterion.
        verbose : bool
            Display intermediate loss values.
        init_detector : dict
            Parameters to initialize the detector.
            (sdr, theta, phi, gamma, bx, by, bz)
        """
        
        # Generate the initial estimate and compute the loss
        est, sdr, rotations, translations = self.project(**init_detector, return_grads=True)
        loss = loss_fn(true_drr, estimate)
        loss.backward(retain_graph=True)
            
        # Start the optimization loop
        for itr in range(n_iters):
            rotations -= alpha1 * rotations.grad
            translations -= alpha2 * translations.grad
            est, rotations, translations = self._project(sdr, rotations, translations)
            loss = loss_fn(true_drr, estimate)
            loss.backward(retain_graph=True)
            
            # Early stopping criterion
            if loss < earlystop:
                break
            if itr % 25 == 0 and verbose:
                print(loss.item())

        return rotations, translations, loss
