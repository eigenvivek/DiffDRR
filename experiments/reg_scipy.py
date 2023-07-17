import csv
import time
from pathlib import Path

import click
import numpy as np
import torch
from scipy.optimize import minimize
from tqdm import tqdm

from diffdrr import DRR, load_example_ct
from diffdrr.metrics import XCorr2


def get_true_drr():
    volume, spacing = load_example_ct()
    bx, by, bz = np.array(volume.shape) * np.array(spacing) / 2
    true_params = {
        "sdr": 200.0,
        "theta": torch.pi,
        "phi": 0,
        "gamma": torch.pi / 2,
        "bx": bx,
        "by": by,
        "bz": bz,
    }
    return volume, spacing, true_params


def get_initial_parameters(true_params):
    sdr = true_params["sdr"]
    theta = true_params["theta"] + np.random.uniform(-np.pi / 4, np.pi / 4)
    phi = true_params["phi"] + np.random.uniform(-np.pi / 3, np.pi / 3)
    gamma = true_params["gamma"] + np.random.uniform(-np.pi / 3, np.pi / 3)
    bx = true_params["bx"] + np.random.uniform(-30.0, 31.0)
    by = true_params["by"] + np.random.uniform(-30.0, 31.0)
    bz = true_params["bz"] + np.random.uniform(-30.0, 31.0)
    return sdr, theta, phi, gamma, bx, by, bz


def run_convergence_exp(
    drr,
    ground_truth,
    true_params,
    filename,
    debug,
):
    def objective_function(geom_params, *optim_params):
        """optim_params = [drr, criterion, ground_truth, sdr]"""
        theta, phi, gamma = geom_params[:3]
        bx, by, bz = geom_params[3:]
        estimate = drr(sdr, theta, phi, gamma, bx, by, bz)
        loss = -criterion(ground_truth, estimate).item()
        t1 = time.perf_counter()
        writer.writerow([t1 - t0, loss, *geom_params])
        return loss

    # Initialize the DRR and optimization parameters
    sdr, theta, phi, gamma, bx, by, bz = get_initial_parameters(true_params)
    criterion = XCorr2(zero_mean_normalized=True)

    with open(filename, "w") as f:

        # Start the training log
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["time", "loss", "theta", "phi", "gamma", "bx", "by", "bz"])
        t0 = time.perf_counter()

        result = minimize(
            objective_function,
            x0=[theta, phi, gamma, bx, by, bz],
            args=[drr, criterion, ground_truth, sdr],
            method="Nelder-Mead",
        )

    if debug:
        print(result.success)


@click.command()
@click.option(
    "-n", "--n_drrs", type=int, default=100, help="Number of DRRs to try to optimize"
)
@click.option(
    "-d", "--debug", is_flag=True, default=False, help="Print debug information"
)
@click.option("--outdir", default="base", type=click.Path())
def main(n_drrs, debug, outdir):

    # Get the ground truth DRR
    volume, spacing, true_params = get_true_drr()
    drr = DRR(volume, spacing, height=100, delx=10.0, device="cuda")
    ground_truth = drr(**true_params)

    # Estimate a random DRR and try to optimize its parameters
    outdir = Path(f"experiments/registration/results/{outdir}/runs")
    outdir.mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(n_drrs)):
        filename = outdir / f"{i}.csv"
        run_convergence_exp(
            drr,
            ground_truth,
            true_params,
            filename,
            debug,
        )


if __name__ == "__main__":
    main()
