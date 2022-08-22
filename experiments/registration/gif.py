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


def is_converged(df):
    return df["loss"].iloc[-1] <= -0.999


def get_fps(df):
    return round(1 / df["time"].mean())


@click.command()
@click.option("-o", "--outdir", type=str, help="Directory with optimization runs.")
@click.option("-n", "--n_gifs", type=int, default=10, help="Number of gifs to make.")
@click.option("--max_n_frames", type=int, default=250, help="Maximum number of frames.")
def main(outdir, n_gifs, max_n_frames):
    csvfiles, converged, not_converged = make_dirs(outdir)
    drr, sdr, ground_truth = make_groundtruth()

    # Find converged and not converged runs
    n_conv = 0
    n_not_conv = 0
    converged_runs, not_converged_runs = [], []
    for csvfile in csvfiles:
        df = pd.read_csv(csvfile)
        fps = get_fps(df)
        if is_converged(df):
            if n_conv >= n_gifs:
                continue
            converged_runs.append([csvfile.stem, df, fps])
            n_conv += 1
        else:
            if n_not_conv >= n_gifs:
                continue
            not_converged_runs.append([csvfile.stem, df, fps])
            n_not_conv += 1
    csvfiles = converged_runs + not_converged_runs

    # Make gifs
    for filename, df, fps in tqdm(csvfiles):
        if len(df) > max_n_frames:
            df = df.iloc[:max_n_frames]
        if is_converged(df):
            outdir = converged
        else:
            outdir = not_converged
        outdir = outdir / f"{filename}.gif"
        animate(df, sdr, drr, ground_truth, verbose=False, out=outdir, fps=fps)


if __name__ == "__main__":
    main()
