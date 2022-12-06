import csv
import time
from pathlib import Path

import click
import numpy as np
import torch
from torch.nn import MSELoss
from tqdm import tqdm

from diffdrr import DRR, load_example_ct
from diffdrr.metrics import geodesic, XCorr2


def get_true_drr():
    """Get parameters for the fixed DRR."""
    volume, spacing = load_example_ct()
    bx, by, bz = torch.tensor(volume.shape) * torch.tensor(spacing) / 2
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
    """Get starting parameters for the moving DRR by perturbing the true params."""
    sdr = true_params["sdr"]
    theta = true_params["theta"] + np.random.uniform(-np.pi / 4, np.pi / 4)
    phi = true_params["phi"] + np.random.uniform(-np.pi / 3, np.pi / 3)
    gamma = true_params["gamma"] + np.random.uniform(-np.pi / 3, np.pi / 3)
    bx = true_params["bx"] + np.random.uniform(-30.0, 31.0)
    by = true_params["by"] + np.random.uniform(-30.0, 31.0)
    bz = true_params["bz"] + np.random.uniform(-30.0, 31.0)
    return sdr, theta, phi, gamma, bx, by, bz


def parse_criterion(criterion):
    """Get the loss function."""
    if criterion == "mse":
        return MSELoss()
    elif criterion == "xcorr2":
        return XCorr2(zero_mean_normalized=True)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def parse_optimizer(optimizer, drr_moving):
    """Get the optimizer."""
    if optimizer == "adam":
        return torch.optim.Adam(
            [
                {"params": [drr_moving.rotations], "lr": 5e-2},
                {"params": [drr_moving.translations], "lr": 5e0},
            ],
        )
    elif optimizer == "sgd":
        return torch.optim.SGD(
            [
                {"params": [drr_moving.rotations], "lr": 5.3e-2},
                {"params": [drr_moving.translations], "lr": 7.5e1},
            ],
            maximize=True,
        )
    elif optimizer == "momentum":
        return torch.optim.SGD(
            [
                {"params": [drr_moving.rotations], "lr": 5.3e-2},
                {"params": [drr_moving.translations], "lr": 7.5e1},
            ],
            momentum=0.9,
            maximize=True,
        )
    elif optimizer == "momentumdampen":
        return torch.optim.SGD(
            [
                {"params": [drr_moving.rotations], "lr": 5.3e-2},
                {"params": [drr_moving.translations], "lr": 7.5e1},
            ],
            momentum=0.9,
            dampening=0.2,
            maximize=True,
        )
    elif optimizer == "lbfgs":
        return torch.optim.LBFGS(
            [drr_moving.rotations, drr_moving.translations],
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


def run_convergence_exp(
    drr_target,
    drr_moving,
    criterion,
    optimizer,
    filename,
    n_itrs,
    debug,
    reg_error_cutoff,
):
    # Get the fixed (ground truth) DRR
    ground_truth = drr_target()

    # Get the loss function and optimizer
    criterion = parse_criterion(criterion)
    optimizer = parse_optimizer(optimizer, drr_moving)

    # Start the optimization log
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            [
                "itr",
                "time",
                "geodesic",
                "loss",
                "theta",
                "phi",
                "gamma",
                "bx",
                "by",
                "bz",
            ]
        )

        # Run the optimization loop
        for itr in range(1, n_itrs + 1):

            # Forward pass: compute the moving DRR
            t0 = time.perf_counter()
            estimate = drr_moving()

            # Compute the loss
            loss = criterion(estimate, ground_truth)
            d_geo = geodesic(drr_moving, drr_target)
            if d_geo < reg_error_cutoff:
                tqdm.write(f"Converged in {itr} iterations")
                break

            # Backward pass: update the 6DoF parameters
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            elapsed = time.perf_counter() - t0

            # Log the results
            trans = [i.item() for i in drr_moving.translations.squeeze()]
            rots = [i.item() for i in drr_moving.rotations.squeeze()]
            writer.writerow([itr, elapsed] + [d_geo.item(), loss.item()] + trans + rots)

            if debug:
                if itr % 25 == 0:
                    tqdm.write(
                        f"{itr}: loss={loss.item():.3f} geodesic={d_geo.item():.3f}"
                    )


@click.command()
@click.option(
    "-n",
    "--n_drrs",
    type=int,
    default=100,
    help="Number of DRRs to try to optimize",
)
@click.option(
    "-i",
    "--n_itrs",
    type=int,
    default=250,
    help="Number of iterations per DRR",
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    default=False,
    help="Print loss and geodesic distance every 25 iterations",
)
@click.option(
    "-c",
    "--criterion",
    type=click.Choice(["mse", "xcorr2"]),
    default="xcorr2",
    help="Loss function to use",
)
@click.option(
    "-o",
    "--optimizer",
    type=click.Choice(["adam", "sgd", "momentum", "momentumdampen", "lbfgs"]),
    default="momentumdampen",
    help="Optimizer to use",
)
@click.option(
    "--reg_error_cutoff",
    type=float,
    default=1.0,
    help="Registration error cutoff (mm)",
)
@click.option(
    "-s",
    "--subsample",
    type=int,
    default=None,
)
@click.option(
    "-r",
    "--reshape",
    type=bool,
    default=True,
)
@click.option(
    "--device",
    default="cuda",
    type=str,
    help="PyTorch device to use",
)
@click.option(
    "--outdir",
    default="base",
    type=click.Path(),
)
def main(
    n_drrs,
    n_itrs,
    debug,
    criterion,
    optimizer,
    reg_error_cutoff,
    subsample,
    reshape,
    device,
    outdir,
):
    print(f"Running {criterion} with {optimizer}")

    # Initialize the fixed (ground truth) DRR
    volume, spacing, true_params = get_true_drr()
    drr_target = DRR(
        volume,
        spacing,
        height=100,
        delx=10.0,
        subsample=subsample,
        reshape=reshape,
        device=device,
    )
    _ = drr_target(**true_params)

    # Initialize the moving DRR
    drr_moving = DRR(
        volume,
        spacing,
        height=100,
        delx=10.0,
        subsample=drr_target.detector.subsample,
        reshape=reshape,
        device=device,
    )

    # Estimate a random DRR and try to optimize its parameters
    outdir = Path(f"experiments/registration/results/{outdir}/runs")
    outdir.mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(n_drrs)):
        # Initialize the moving DRR
        sdr, theta, phi, gamma, bx, by, bz = get_initial_parameters(true_params)
        _ = drr_moving(sdr, theta, phi, gamma, bx, by, bz)

        # Run the optimization
        filename = outdir / f"{i}.csv"
        run_convergence_exp(
            drr_target,
            drr_moving,
            criterion,
            optimizer,
            filename,
            n_itrs,
            debug,
            reg_error_cutoff,
        )


if __name__ == "__main__":
    main()
