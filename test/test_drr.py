import numpy as np

from diffdrr import DRR, load_example_ct


def test_DRR():
    volume, spacing = load_example_ct()
    bx, by, bz = np.array(volume.shape) * np.array(spacing) / 2
    detector_kwargs = {
        "sdr": 300.0,
        "theta": np.pi,
        "phi": 0,
        "gamma": np.pi / 2,
        "bx": bx,
        "by": by,
        "bz": bz,
    }
    try:
        drr = DRR(volume, spacing, height=200, delx=1.4e-2, device="cuda")
    except ValueError:
        drr = DRR(volume, spacing, height=200, delx=1.4e-2, device="cpu")
    img = drr(**detector_kwargs)
    assert img.shape == (1, 200, 200)
