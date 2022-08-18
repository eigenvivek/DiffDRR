import csv
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src import DRR, read_dicom
from src.visualization import plot_drr


def make_ground_truth():
    volume, spacing = read_dicom("data/cxr/")
    bx, by, bz = np.array(volume.shape) * np.array(spacing) / 2
    true_params = torch.tensor([torch.pi, 0, torch.pi / 2, bx, by, bz])
    return volume, spacing, true_params


def get_perturbation():
    del_t = np.random.uniform(-torch.pi, torch.pi)
    del_p = np.random.uniform(0, 2 * torch.pi)
    del_g = np.random.uniform(-torch.pi / 2, 3 * torch.pi / 2)
    del_x = np.random.uniform(-5, 5)
    del_y = np.random.uniform(-5, 5)
    del_z = np.random.uniform(-5, 5)
    return torch.tensor([del_t, del_p, del_g, del_x, del_y, del_z])


def make_image(drr, true_params):
    perturbed_params = true_params + get_perturbation()
    perturbed = drr(200.0, *perturbed_params)
    return perturbed_params, perturbed


@click.command()
@click.option("-n", "--n_drrs", default=100, help="Number of DRRs to generate")
@click.option("-o", "--outdir", default="dataset/", help="Output directory")
def main(n_drrs, outdir):
    volume, spacing, true_params = make_ground_truth()
    drr = DRR(volume, spacing, height=100, delx=5e-2, device="cuda")

    outdir = Path(f"experiments/initialization/{outdir}")
    ptdir = outdir / "drrs"
    ptdir.mkdir(parents=True, exist_ok=True)
    filename = outdir / "data.csv"
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["img", "sdr", "theta", "phi", "gamma", "bx", "by", "bz"])

        for i in tqdm(range(n_drrs)):
            img_name = ptdir / f"{i}.pt"
            perturbed_params, perturbed = make_image(drr, true_params)
            torch.save(perturbed, img_name)
            writer.writerow([img_name, 200.0, *perturbed_params.numpy()])


if __name__ == "__main__":
    main()
