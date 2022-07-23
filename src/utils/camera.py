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
        source, center, u, v = _get_basis(self.sdr, self.angles)
        source += self.translation
        center += self.translation

        # Construt the detector plane
        t = (torch.arange(-height // 2, height // 2, device=self.device) + 1) * delx
        s = (torch.arange(-width // 2, width // 2, device=self.device) + 1) * dely
        coefs = torch.cartesian_prod(t, s).reshape(height, width, 2)
        rays = coefs @ torch.stack([u, v])
        rays += center
        return source, center, rays

    
def _get_basis(rho, angles):
    # Get the rotation of 3D space
    R = rho * Rxyz(angles)
    
    # Get the detector center and X-ray source
    source = R[:, 0]
    center = -source
    
    # Get the basis of the detector plane (before translation)
    u, v = R[:, 1], R[:, 2]
    u = u / torch.norm(u)
    v = v / torch.norm(v)
    
    return source, center, u, v


# Define 3D rotation matrices
def Rxyz(angles):
    theta, phi, gamma = angles
    return Rz(theta) @ Ry(phi) @ Rx(gamma)


def Rx(theta):
    return torch.tensor([
        [1, 0               ,  0],
        [0, torch.cos(theta), -torch.sin(theta)],
        [0, torch.sin(theta),  torch.cos(theta)]
    ])


def Ry(theta):
    return torch.tensor([
        [ torch.cos(theta),  0, torch.sin(theta)],
        [0                ,  1, 0],
        [-torch.sin(theta),  0, torch.cos(theta)]
    ])


def Rz(theta):
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta),  torch.cos(theta), 0],
        [0               ,  0               , 1]
    ])
