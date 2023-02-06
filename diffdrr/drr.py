import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .projectors.siddon import Siddon
from .utils import reshape_subsampled_drr
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
        dtype=None,
        device=None,
        params=None,
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
        params : torch.Tensor, optional
            The parameters of the camera, including SDR, rotations, and translations.
            If provided, the DRR module will be initialized in optimization mode.
            Otherwise, the DRR module will be in rendering mode and the viewing angle
            must be provided at each forward pass.

            Note that this also enables batch rendering of DRRs!
        """
        super().__init__()

        # Initialize the volume
        if params is not None:
            self.optimization_mode = True
            self.sdr = nn.Parameter(params[..., 0:1])
            self.rotations = nn.Parameter(params[..., 1:4])
            self.translations = nn.Parameter(params[..., 4:7])
        else:
            self.optimization_mode = False

        # Initialize the X-ray detector
        width = height if width is None else width
        dely = delx if dely is None else dely
        self.detector = Detector(
            height,
            width,
            delx,
            dely,
            n_subsample=int(height * width * p_subsample)
            if p_subsample is not None
            else None,
        )

        # Initialize the Projector and register its parameters
        self.spacing = nn.Parameter(torch.tensor(spacing), requires_grad=False)
        self.volume = nn.Parameter(torch.tensor(volume).flip([0]), requires_grad=False)
        assert projector != "siddon_jacobs", "Siddon-Jacobs projector is deprecated."
        if projector == "siddon":
            self.siddon = Siddon(self.volume, self.spacing)
        else:
            raise ValueError("Invalid projector type.")
        self.reshape = reshape

        # Dummy tensor for device and dtype
        self.dummy = nn.Parameter(
            torch.tensor([0.0]),
            requires_grad=False,
        )
        if dtype is not None or device is not None:
            raise DeprecationWarning(
                """
                dtype and device are deprecated. 
                Instead, use .to(dtype) or .to(device) to update the DRR module.
                """
            )

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
        Generate a DRR from a particular viewing angle. If the DRR module is in
        optimization mode, the viewing angle is ignored and the DRR is generated
        from the provided parameters.

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
        if not self.optimization_mode:
            self.sdr = torch.tensor([[sdr]]).to(self.dummy)
            self.rotations = torch.tensor([[theta, phi, gamma]]).to(self.dummy)
            self.translations = torch.tensor([[bx, by, bz]]).to(self.dummy)
        params = [sdr, theta, phi, gamma, bx, by, bz]
        if any(arg is not None for arg in params) and self.optimization_mode:
            raise ValueError("Cannot provide parameters in optimization mode.")

        source, target = self.detector.make_xrays(
            sdr=self.sdr,
            rotations=self.rotations,
            translations=self.translations,
        )
        img = self.siddon.raytrace(source, target)

        if self.reshape:
            if self.detector.n_subsample is None:
                img = img.view(-1, 1, self.detector.height, self.detector.width)
            else:
                img = reshape_subsampled_drr(img, self.detector, len(target))
        return img

    def plot_geometry(self, ax=None):
        """Visualize the geometry of the detector."""
        if len(list(self.parameters())) == 0:
            raise ValueError("Parameters uninitialized.")
        source, target = self.detector.make_xrays(
            sdr=self.sdr,
            rotations=self.rotations,
            translations=self.translations,
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
