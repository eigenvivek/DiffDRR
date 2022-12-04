import csv
import time
from pathlib import Path

import click
import numpy as np
import torch
from tqdm import tqdm

from diffdrr import load_example_ct, DRR


def get_ground_truth():
    # Make the ground truth DRR
    volume, spacing = load_example_ct()
    drr = DRR(volume, spacing, height=100, delx=10.0, device="cuda")
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
    ground_truth = drr(**true_params)
    return ground_truth, true_params, drr


def perturb(true_params, param, drr):
    if param == "theta":
        perturbation = np.random.uniform(-np.pi / 3, np.pi / 3)
    elif param == "phi":
        perturbation = np.random.uniform(-np.pi / 3, np.pi / 3)
    elif param == "gamma":
        perturbation = np.random.uniform(-np.pi / 3, np.pi / 3)
    elif param == "bx":
        perturbation = np.random.uniform(-100.0, 101.0)
    elif param == "by":
        perturbation = np.random.uniform(-100.0, 101.0)
    elif param == "bz":
        perturbation = np.random.uniform(-100.0, 101.0)
    else:
        raise ValueError(f"Unknown parameter: {param}")
    perturbed_params = true_params.copy()
    perturbed_params[param] += perturbation
    return drr(**perturbed_params), true_params[param], perturbed_params[param]


def finite_diff(loss, true_param, perturbed_param):
    t0 = time.perf_counter()
    grad = loss.item() / (true_param - perturbed_param)
    t1 = time.perf_counter()
    return grad, t1 - t0


def auto_diff(loss, drr, param):
    loss.backward(retain_graph=True)
    t0 = time.perf_counter()
    if param == "theta":
        grad = drr.rotations.grad[0]
    elif param == "phi":
        grad = drr.rotations.grad[1]
    elif param == "gamma":
        grad = drr.rotations.grad[2]
    elif param == "bx":
        grad = drr.translations.grad[0]
    elif param == "by":
        grad = drr.translations.grad[1]
    elif param == "bz":
        grad = drr.translations.grad[2]
    else:
        raise ValueError(f"Unknown parameter: {param}")
    t1 = time.perf_counter()
    return grad.item(), t1 - t0


def run(true_params, param, drr, ground_truth):
    perturbed_drr, true_param, perturbed_param = perturb(true_params, param, drr)
    loss = torch.norm(ground_truth - perturbed_drr)
    fin_grad, fin_runtime = finite_diff(loss, true_param, perturbed_param)
    auto_grad, auto_runtime = auto_diff(loss, drr, param)
    return true_param, perturbed_param, fin_grad, fin_runtime, auto_grad, auto_runtime


@click.command()
@click.option("-n", "--n_runs", type=int, help="Number of runs for each parameter")
@click.option("-o", "--outfile", type=click.Path(), help="Outfile to save results to")
def main(n_runs, outfile):
    ground_truth, true_params, drr = get_ground_truth()
    params = ["theta", "phi", "gamma", "bx", "by", "bz"]

    currdir = Path(__file__).parent
    outfile = currdir / f"{outfile}.csv"
    with open(outfile, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            [
                "param",
                "true_param",
                "perturbed_param",
                "finite_diff",
                "finite_diff_time",
                "auto_diff",
                "auto_diff_time",
            ]
        )
        for param in params:
            for _ in tqdm(range(n_runs), desc=param):
                exp_result = run(true_params, param, drr, ground_truth)
                writer.writerow([param, *exp_result])


if __name__ == "__main__":
    main()
