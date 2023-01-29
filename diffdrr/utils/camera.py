import torch
from torch.nn.functional import normalize

from .backend import get_device


class Detector:
    """
    Construct a 6 DoF X-ray detector system. This model is based on a C-Arm.

    Inputs
    ------
    height : int
        Height of the X-ray detector (ie, DRR height)
    width : int
        Width of the X-ray detector (ie, DRR width)
    delx : float
        Pixel spacing in the X-direction of the X-ray detector
    dely : float
        Pixel spacing in the Y-direction of the X-ray detector
    n_subsample : int
        Number of target points to randomly sample
    device : str or torch.device
        Compute device. If str, either "cpu", "cuda", or "mps".
    """

    def __init__(self, height, width, delx, dely, n_subsample, dtype, device):
        self.height = height
        self.width = width
        self.delx = delx
        self.dely = dely
        self.dtype = dtype
        self.device = device if isinstance(device, torch.device) else get_device(device)
        self.n_subsample = n_subsample
        if self.n_subsample is not None:
            self.subsamples = []

    def make_xrays(self, sdr, rotations, translations):
        """
        Inputs
        ------
        sdr : torch.Tensor
            Source-to-Detector radius (half of the Source-to-Detector distance)
        rotations : torch.Tensor
            Vector of C-Arm rotations (theta, phi, gamma) for azimuthal, polar, and roll
        translations : torch.Tensor
            Vector of volume translations (bx, by, bz)
        """

        # Get the detector plane normal vector
        assert len(rotations) == len(translations)
        source, center, basis = _get_basis(sdr, rotations, self.device)
        source += translations.unsqueeze(1)
        center += translations.unsqueeze(1)

        # Construct the detector plane
        h_off = 1.0 if self.height % 2 else 0.5  # Different offsets for even or odd heights
        w_off = 1.0 if self.width % 2 else 0.5
        t = (torch.arange(-self.height // 2, self.height // 2) + h_off) * self.delx
        s = (torch.arange(-self.width // 2, self.width // 2) + w_off) * self.dely

        coefs = torch.cartesian_prod(t, s).reshape(-1, 2).to(self.device).to(self.dtype)
        target = torch.einsum("bcd,nc->bnd", basis, coefs)
        target += center
        if self.n_subsample is not None:
            sample = torch.randperm(self.height * self.width)[: int(self.n_subsample)]
            target = target[:, sample, :]
            self.subsamples.append(sample.tolist())
        return source, target


def _get_basis(rho, rotations, device):
    # Get the rotation of 3D space
    R = rho * Rxyz(rotations, device)

    # Get the detector center and X-ray source
    source = R[..., 0].unsqueeze(1)
    center = -source

    # Get the basis of the detector plane (before translation)
    R_ = normalize(R.clone(), dim=-1)
    u, v = R_[..., 1], R_[..., 2]
    basis = torch.stack([u, v], dim=1)

    return source, center, basis


# Define 3D rotation matrices
def Rxyz(rotations, device):
    theta, phi, gamma = rotations[:, 0], rotations[:, 1], rotations[:, 2]
    batch_size = len(rotations)
    R_z = Rz(theta, batch_size, device)
    R_y = Ry(phi, batch_size, device)
    R_x = Rx(gamma, batch_size, device)
    return torch.einsum("bij,bjk,bkl->bil", R_z, R_y, R_x)


def Rx(gamma, batch_size, device):
    t0 = torch.zeros(batch_size, 1, device=device)
    t1 = torch.ones(batch_size, 1, device=device)
    return torch.stack(
        [
            t1,
            t0,
            t0,
            t0,
            torch.cos(gamma.unsqueeze(1)),
            -torch.sin(gamma.unsqueeze(1)),
            t0,
            torch.sin(gamma.unsqueeze(1)),
            torch.cos(gamma.unsqueeze(1)),
        ],
        dim=1,
    ).reshape(batch_size, 3, 3)


def Ry(phi, batch_size, device):
    t0 = torch.zeros(batch_size, 1, device=device)
    t1 = torch.ones(batch_size, 1, device=device)
    return torch.stack(
        [
            torch.cos(phi.unsqueeze(1)),
            t0,
            torch.sin(phi.unsqueeze(1)),
            t0,
            t1,
            t0,
            -torch.sin(phi.unsqueeze(1)),
            t0,
            torch.cos(phi.unsqueeze(1)),
        ]
    ).reshape(batch_size, 3, 3)


def Rz(theta, batch_size, device):
    t0 = torch.zeros(batch_size, 1, device=device)
    t1 = torch.ones(batch_size, 1, device=device)
    return torch.stack(
        [
            torch.cos(theta.unsqueeze(1)),
            -torch.sin(theta.unsqueeze(1)),
            t0,
            torch.sin(theta.unsqueeze(1)),
            torch.cos(theta.unsqueeze(1)),
            t0,
            t0,
            t0,
            t1,
        ]
    ).reshape(batch_size, 3, 3)
