import torch


class Detector:
    """
    Construct a 6 DoF X-ray detector system. This model is based on a C-Arm.

    Inputs
    ------
        sdr        : Source-to-Detector radius (half of the Source-to-Detector distance)
        theta      : azimuthal angle
        phi        : polar angle
        gamma      : rotation of detector plane
        bx, by, bz : translation of the volume
    """

    def __init__(self, sdr, theta, phi, gamma, bx, by, bz, device):
        self.sdr = torch.tensor([sdr], dtype=torch.float32, device=device)
        self.angles = torch.tensor([theta, phi, gamma], dtype=torch.float32, device=device, requires_grad=True)
        self.translation = torch.tensor([bx, by, bz], dtype=torch.float32, device=device, requires_grad=True)
        self.device = device

    def make_xrays(self, height, width, delx, dely):

        # Get the detector plane normal vector
        source, center, u, v = _get_basis(self.sdr, self.angles, self.device)
        source += self.translation
        center += self.translation

        # Construt the detector plane
        t = (torch.arange(-height // 2, height // 2, device=self.device) + 1) * delx
        s = (torch.arange(-width // 2, width // 2, device=self.device) + 1) * dely
        coefs = torch.cartesian_prod(t, s).reshape(height, width, 2)
        rays = coefs @ torch.stack([u, v])
        rays += center
        return source, rays


def _get_basis(rho, angles, device):
    # Get the rotation of 3D space
    R = rho * Rxyz(angles, device)

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
def Rxyz(angles, device):
    theta, phi, gamma = angles
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
