from pathlib import Path

import numpy as np
from pydicom import dcmread


def read_dicom(dcmdir="data/cxr"):

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

    delZs = np.diff(delZs)
    delZ = np.abs(np.unique(delZs)[0])
    spacing = [delX, delY, delZ]

    return volume, spacing
