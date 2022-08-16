from pathlib import Path

import numpy as np
import pandas as pd

from src import read_dicom, DRR
from src.visualization import animate


def make_dirs():
    currdir = Path(__file__).parent
    csvfiles = (currdir / "tmp").glob("*.csv")
    converged = currdir / "gif/converged/"
    not_converged = currdir / "gif/not_converged/"
    converged.mkdir(parents=True, exist_ok=True)
    not_converged.mkdir(parents=True, exist_ok=True)
    return csvfiles, converged, not_converged


def make_groundtruth():
    volume, spacing = read_dicom("data/cxr")
    bx, by, bz = np.array(volume.shape) * np.array(spacing) / 2
    drr = DRR(volume, spacing, height=100, delx=5e-2, device="cpu")
    sdr = 200.0
    ground_truth = drr(sdr, np.pi, 0, np.pi / 2, bx, by, bz)
    return drr, sdr, ground_truth


def main():
    csvfiles, converged, not_converged = make_dirs()
    drr, sdr, ground_truth = make_groundtruth()
    for csvfile in csvfiles:
        df = pd.read_csv(csvfile)
        if len(df) < 500:
            outdir = converged
        else:
            outdir = not_converged
        outdir = outdir / f"{csvfile.stem}.gif"
        animate(df[::10], sdr, drr, ground_truth, out=outdir)


if __name__ == "__main__":
    main()
