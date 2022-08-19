import torch

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
    device : str or torch.device
        Compute device. If str, either "cpu", "cuda", or "mps".
    """

    def __init__(self, height, width, delx, dely, device):
        self.height = height
        self.width = width
        self.delx = delx
        self.dely = dely
        self.device = device if isinstance(device, torch.device) else get_device(device)

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
        source, center, u, v = _get_basis(sdr, rotations, self.device)
        source += translations
        center += translations

        # Construt the detector plane
        t = (
            torch.arange(-self.height // 2, self.height // 2, device=self.device) + 1
        ) * self.delx
        s = (
            torch.arange(-self.width // 2, self.width // 2, device=self.device) + 1
        ) * self.dely
        coefs = torch.cartesian_prod(t, s).reshape(self.height, self.width, 2)
        rays = coefs @ torch.stack([u, v])
        rays += center
        return source, rays


def _get_basis(rho, rotations, device):
    # Get the rotation of 3D space
    R = rho * Rxyz(rotations, device)

    # Get the detector center and X-ray source
    source = R[:, 0]
    center = -source

    # Get the basis of the detector plane (before translation)
    # TODO: normalizing the vectors seems to break the gradient, fix in future
    u, v = R[:, 1], R[:, 2]
    # u_ = u / torch.norm(u)
    # v_ = v / torch.norm(v)

    return source, center, u, v


# Define 3D rotation matrices
def Rxyz(rotations, device):
    theta, phi, gamma = rotations
    return Rz(theta, device) @ Ry(phi, device) @ Rx(gamma, device)


def Rx(theta, device):
    t0 = torch.zeros(1, device=device)[0]
    t1 = torch.ones(1, device=device)[0]
    return torch.stack(
        [
            t1,
            t0,
            t0,
            t0,
            torch.cos(theta),
            -torch.sin(theta),
            t0,
            torch.sin(theta),
            torch.cos(theta),
        ]
    ).reshape(3, 3)


def Ry(theta, device):
    t0 = torch.zeros(1, device=device)[0]
    t1 = torch.ones(1, device=device)[0]
    return torch.stack(
        [
            torch.cos(theta),
            t0,
            torch.sin(theta),
            t0,
            t1,
            t0,
            -torch.sin(theta),
            t0,
            torch.cos(theta),
        ]
    ).reshape(3, 3)


def Rz(theta, device):
    t0 = torch.zeros(1, device=device)[0]
    t1 = torch.ones(1, device=device)[0]
    return torch.stack(
        [
            torch.cos(theta),
            -torch.sin(theta),
            t0,
            torch.sin(theta),
            torch.cos(theta),
            t0,
            t0,
            t0,
            t1,
        ]
    ).reshape(3, 3)
