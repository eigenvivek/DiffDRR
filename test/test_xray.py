import torch

from diffdrr import Detector


def test_Detector():
    try:
        detector = Detector(
            100,
            100,
            7.5e-2,
            7.5e-2,
            n_subsample=None,
            dtype=torch.float32,
            device="cuda",
        )
    except ValueError:
        detector = Detector(
            100,
            100,
            7.5e-2,
            7.5e-2,
            n_subsample=None,
            dtype=torch.float32,
            device="cpu",
        )
    sdr = torch.tensor([200.0]).to(detector.device)
    rotations = torch.rand(4, 3).to(detector.device)
    translations = torch.rand(4, 3).to(detector.device)
    source, target = detector.make_xrays(sdr, rotations, translations)
    assert source.shape == (4, 1, 3)
    assert target.shape == (4, 100 * 100, 3)
