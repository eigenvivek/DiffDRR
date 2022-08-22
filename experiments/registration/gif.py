from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import DRR, load_example_ct
from src.visualization import animate


def make_dirs(outdir):
    currdir = Path(__file__).parent
    csvfiles = (currdir / "results" / outdir / "runs").glob("*.csv")
    converged = currdir / f"results/{outdir}/gifs/converged/"
    converged.mkdir(parents=True, exist_ok=True)
    not_converged = currdir / f"results/{outdir}/gifs/not_converged/"
    not_converged.mkdir(parents=True, exist_ok=True)
    return csvfiles, converged, not_converged


def make_groundtruth():
    volume, spacing = load_example_ct()
    bx, by, bz = np.array(volume.shape) * np.array(spacing) / 2
    drr = DRR(volume, spacing, height=100, delx=5e-2, device="cuda")
    sdr = 200.0
    ground_truth = drr(sdr, np.pi, 0, np.pi / 2, bx, by, bz)
    return drr, sdr, ground_truth


@click.command()
@click.command("-n", "--n_gifs", type=int, default=10, help="Number of gifs to make.")
@click.option("--outdir", type=str, help="Directory with optimization runs.")
def main(n_gifs, outdir):
    csvfiles, converged, not_converged = make_dirs(outdir)
    csvfiles = list(csvfiles)
    drr, sdr, ground_truth = make_groundtruth()

    n_conv = 0
    n_not_conv = 0
    for csvfile in tqdm(csvfiles):
        df = pd.read_csv(csvfile)
        if df["loss"].iloc[-1] <= -0.999:
            if n_conv >= n_gifs:
                continue
            outdir = converged
            n_conv += 1
        else:
            if n_not_conv >= n_gifs:
                continue
            outdir = not_converged
            n_not_conv += 1
        outdir = outdir / f"{csvfile.stem}.gif"
        animate(df, sdr, drr, ground_truth, verbose=False, out=outdir)


if __name__ == "__main__":
    main()
