import torch


class Detector:
    def __init__(self, source, center, height, width, delx, dely, device):
        self.source = torch.tensor(source, device=device, requires_grad=True)
        self.center = torch.tensor(center, device=device, requires_grad=True)
        self.height = height
        self.width = width
        self.delx = delx
        self.dely = dely
        self.device = device

    def make_xrays(self):

        # Get the detector plane normal vector
        normal = self.source - self.center
        normal = normal / torch.norm(normal)
        u, v, w = _get_basis(normal)
        assert torch.allclose(w, normal)

        # Construt the detector plane
        t = (torch.arange(-self.height // 2, self.height // 2, device=self.device) + 1) * self.delx
        s = (torch.arange(-self.width // 2, self.width // 2, device=self.device) + 1) * self.dely
        coefs = torch.cartesian_prod(t, s).reshape(self.height, self.width, 2)
        targets = coefs @ torch.stack([u, v])
        targets += self.center
        return targets


def _get_basis(w):
    t = _get_noncollinear_vector(w)
    t = t / torch.norm(t)
    u = torch.cross(t, w)
    v = torch.cross(u, w)
    return u, v, w


def _get_noncollinear_vector(w):
    t = w.clone()
    i = torch.argmin(torch.abs(w))
    t[i] = 1
    return t