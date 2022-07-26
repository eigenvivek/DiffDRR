from .projectors.siddon import Siddon
from .utils.backend import get_device
from .utils.camera import Detector


def make_drr(
    volume,
    spacing,
    detector_kwargs,
    height,
    delx,
    width=None,
    dely=None,
    device="cpu",
    return_grads=False,
):
    """
    Compute a DRR from a volume.

    Parameters
    ----------
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
    device : str
        Compute devicee, either "cpu", "cuda", or "mps".
    return_grads : bool, optional
        If True, return
    detector_kwargs : dict
        Keyword arguments for the detector:
            sdr   : float
            theta : float
            phi   : float
            gamma : float
            bx    : float
            by    : float
            bz    : float

    Returns
    -------
    drr : torch.Tensor
        The DRR.
    """

    # Get the device
    device = get_device(device)

    # Make the detector plane
    if width is None:
        width = height
    if dely is None:
        dely = delx
    detector = Detector(device=device, **detector_kwargs)
    source, rays = detector.make_xrays(height, width, delx, dely)

    # Make the DRR
    siddon = Siddon(volume, spacing, device)
    drr = siddon.raytrace(source, rays)
    if return_grads is False:
        return drr
    else:
        return drr, detector.angles, detector.translation
