import torch
from torch.nn.functional import normalize

from ..utils.camera import Rxyz


def geodesic(drr1, drr2, return_components=False):
    """Calculate the geodesic distance between two 6DoF projectors."""
    assert drr1.sdr == drr2.sdr
    assert drr1.device == drr2.device

    # Translational distance
    d_translation = torch.linalg.norm(drr1.translations.data - drr2.translations.data)

    # Rotational distance
    R1 = Rxyz(drr1.rotations, drr1.device).squeeze()
    R2 = Rxyz(drr2.rotations, drr2.device).squeeze()
    ones = normalize(torch.ones(1, 3)).squeeze().to(drr1.device) * drr1.sdr
    rotated = (R1 @ ones) @ (R2 @ ones) / drr1.sdr.pow(2)
    d_rotation = drr1.sdr * torch.arccos(rotated)

    if return_components:
        return d_translation, d_rotation
    else:
        return d_translation + d_rotation
