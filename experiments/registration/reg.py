import csv
import time
from pathlib import Path

import click
import numpy as np
import torch
from tqdm import tqdm

from src import read_dicom, DRR
from src.metrics import XCorr2


def get_true_drr():
    volume, spacing = read_dicom("data/cxr")
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
    n_itrs, drr, ground_truth, true_params, filename, debug,
    lr_rotations, lr_translations, momentum, dampening
):
    # Initialize the DRR and optimization parameters
    sdr, theta, phi, gamma, bx, by, bz = get_initial_parameters(true_params)
    _ = drr(sdr, theta, phi, gamma, bx, by, bz)  # Initialize the DRR generator
    criterion = XCorr2(zero_mean_normalized=True)
    optimizer = torch.optim.SGD(
        [{"params": [drr.rotations], "lr": lr_rotations}, {"params": [drr.translations], "lr": lr_translations}],
        momentum=momentum,
        dampening=dampening,
    )

    with open(filename, "w") as f:

        # Start the training log
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["itr", "time", "loss", "theta", "phi", "gamma", "bx", "by", "bz"])
        t0 = time.perf_counter()

        # Start the training loop
        for itr in range(n_itrs):

            # Forward pass: compute the predicted DRR
            estimate = drr()
            t1 = time.perf_counter()

            # Compute the negative ZNCC as loss
            loss = -criterion(ground_truth, estimate)
            theta, phi, gamma = drr.rotations
            bx, by, bz = drr.translations
            writer.writerow(
                [itr, t1 - t0] + [i.item() for i in [loss, theta, phi, gamma, bx, by, bz]]
            )

            if debug:
                if itr % 25 == 0:
                    print(itr, loss.item())
            if loss < -0.999:
                tqdm.write(f"Converged in {itr} iterations")
                break

            # Backward pass: update the weights
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            t0 = t1


@click.command()
@click.option("-n", "--n_drrs", type=int, default=100, help="Number of DRRs to try to optimize")
@click.option("-i", "--n_itrs", type=int, default=500, help="Number of iterations per DRR")
@click.option("-d", "--debug", is_flag=True, default=False, help="Print debug information")
@click.option("--lr_rotations", type=float, default=5.3e-2, help="Rotational learning rate")
@click.option("--lr_translations", type=float, default=7.5e1, help="Translations learning rate")
@click.option("--momentum", type=float, default=0.9, help="SGD momentum factor")
@click.option("--dampening", type=float, default=0.2, help="SGD dampening factor for momentum")
@click.option("--outdir", default="base", type=click.Path())
def main(n_drrs, n_itrs, debug, lr_rotations, lr_translations, momentum, dampening, outdir):

    # Get the ground truth DRR
    volume, spacing, true_params = get_true_drr()
    drr = DRR(volume, spacing, height=100, delx=5e-2, device="cuda")
    ground_truth = drr(**true_params)

    # Estimate a random DRR and try to optimize its parameters
    outdir = Path(f"experiments/registration/results/{outdir}")
    outdir.mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(n_drrs)):
        filename = f"{outdir}/{i}.csv"
        run_convergence_exp(
            n_itrs, drr, ground_truth, true_params, filename, debug,
            lr_rotations, lr_translations, momentum, dampening
        )


if __name__ == "__main__":
    main()
