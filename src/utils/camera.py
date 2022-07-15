from itertools import product

import torch


class Detector:
    def __init__(self, source, center, height, width, delx, dely):
        self.source = torch.tensor(source, requires_grad=True)
        self.center = torch.tensor(center, requires_grad=True)
        self.height = height
        self.width = width
        self.delx = delx
        self.dely = dely

    def make_xrays(self):

        # Get the detector plane normal vector
        normal = self.source - self.center
        normal = normal / torch.norm(normal)
        u, v, w = _get_basis(normal)
        assert torch.allclose(w, normal)

        # Construt the detector plane
        t = (torch.arange(-self.height // 2, self.height // 2) + 1) * self.delx
        s = (torch.arange(-self.width // 2, self.width // 2) + 1) * self.dely
        t = t.repeat(self.height, 1)
        s = s.repeat(self.width, 1).T
        coefs = torch.stack([t, s])
        coefs = coefs.transpose(2, 0)
        targets = coefs @ torch.stack([u, v])
        targets += self.center
        targets = targets.transpose(0, 2)
        return targets


def _get_basis(normal):
    w = normal / torch.norm(normal)
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
