import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .projectors.siddon import Siddon
from .utils import reshape_subsampled_drr
from .utils.backend import get_device
from .utils.camera import Detector
from .visualization import plot_camera, plot_volume


class DRR(nn.Module):
    def __init__(
        self,
        volume,
        spacing,
        height,
        delx,
        width=None,
        dely=None,
        projector="siddon",
        p_subsample=None,
        reshape=True,
        dtype=torch.float32,
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
        projector : str, optional
            The type of projector, either "siddon" or "siddon_jacobs".
        p_subsample : int, optional
            Proportion of target points to randomly sample for each forward pass
        reshape : bool, optional
            If True, return DRR as (b, n1, n2) tensor. If False, return as (b, n) tensor.
        dtype : torch.dtype, optional
            The data type of the DRR, default=torch.float32
        device : str
            Compute device, either "cpu", "cuda", or "mps".
        """
        super().__init__()
        self.dtype = dtype
        self.device = get_device(device)

        # Initialize the X-ray detector
        width = height if width is None else width
        dely = delx if dely is None else dely
        n_subsample = (
            int(height * width * p_subsample) if p_subsample is not None else None
        )
        self.detector = Detector(
            height,
            width,
            delx,
            dely,
            n_subsample,
            self.dtype,
            self.device,
        )

        # Initialize the Projector and register its parameters
        if projector == "siddon":
            self.siddon = Siddon(volume, spacing, self.dtype, self.device)
        elif projector == "siddon_jacobs":
            raise DeprecationWarning(
                "Siddon-Jacobs projector is deprecated and does not work in this version."
            )
        else:
            raise ValueError("Invalid projector type.")
        self.register_parameter("sdr", None)
        self.register_parameter("rotations", None)
        self.register_parameter("translations", None)

        self.reshape = reshape

    def forward(
        self,
        sdr=None,
        theta=None,
        phi=None,
        gamma=None,
        bx=None,
        by=None,
        bz=None,
    ):
        """
        Generate a DRR from a particular viewing angle.

        Pass projector parameters to initialize a new viewing angle.
        If uninitialized, the model will not run.
        """
        params = [sdr, theta, phi, gamma, bx, by, bz]
        if any(arg is not None for arg in params):
            self.initialize_parameters(*params)
        source, target = self.detector.make_xrays(
            self.sdr,
            self.rotations,
            self.translations,
        )
        img = self.siddon.raytrace(source, target)
        if self.reshape:
            if self.detector.n_subsample is None:
                img = img.view(-1, self.detector.height, self.detector.width)
            else:
                img = reshape_subsampled_drr(img, self.detector, len(target))
        return img

    def initialize_parameters(self, sdr, theta, phi, gamma, bx, by, bz):
        """
        Set the initial parameters for generating a DRR.

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
        """
        tensor_args = {"dtype": self.dtype, "device": self.device}
        # Assume that SDR is given for a 6DoF registration problem
        self.sdr = nn.Parameter(torch.tensor([sdr], **tensor_args), requires_grad=False)
        self.rotations = nn.Parameter(
            torch.tensor([[theta, phi, gamma]], **tensor_args)
        )
        self.translations = nn.Parameter(torch.tensor([[bx, by, bz]], **tensor_args))

    def plot_geometry(self, ax=None):
        """Visualize the geometry of the detector."""
        if len(list(self.parameters())) == 0:
            raise ValueError("Parameters uninitialized.")
        source, target = self.detector.make_xrays(
            self.sdr,
            self.rotations,
            self.translations,
        )
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        ax = plot_camera(source, target, ax)
        ax = plot_volume(
            np.array(self.siddon.volume.detach().cpu()),
            np.array(self.siddon.spacing.detach().cpu()),
            *self.translations.detach().cpu().numpy(),
            ax=ax,
        )

    def __repr__(self):
        params = [str(param) for param in self.parameters()]
        if len(params) == 0:
            return "Parameters uninitialized."
        else:
            return "\n".join(params)
