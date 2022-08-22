from pathlib import Path

import numpy as np
from pydicom import dcmread


def read_dicom(dcmdir="data/cxr", correct_zero=True):
    """
    Inputs
    -----
    dcmdir : Path or str
        Path to a DICOM directory
    correct_zero : bool
        Make 0 the minimum value the CT

    Returns
    -------
    volume : ndarray
        3D array containing voxels of imaging data
    spacing : list
        X-, Y-, and Z-directional voxel spacings
    """

    dcmfiles = Path(dcmdir).glob("*.dcm")
    dcmfiles = list(dcmfiles)
    dcmfiles.sort()
    ds = dcmread(dcmfiles[0])

    nx, ny = ds.pixel_array.shape
    nz = len(dcmfiles)
    delX, delY = ds.PixelSpacing
    delX, delY = float(delX), float(delY)
    volume = np.zeros((nx, ny, nz))

    delZs = []
    for idx, dcm in enumerate(dcmfiles):
        ds = dcmread(dcm)
        volume[:, :, idx] = ds.pixel_array
        delZs.append(ds.ImagePositionPatient[2])

    if correct_zero:
        volume[volume == volume.min()] = 0.0

    delZs = np.diff(delZs)
    delZ = np.abs(np.unique(delZs)[0])
    spacing = [delX, delY, delZ]

    return volume, spacing


def load_example_ct():
    """Load an example chest CT for demonstration purposes."""
    currdir = Path(__file__).resolve().parent
    dcmdir = currdir / "cxr"
    return read_dicom(dcmdir)
