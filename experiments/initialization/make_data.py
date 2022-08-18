import csv
from pathlib import Path

import click
import numpy as np
import torch
from tqdm import tqdm

from src import DRR, read_dicom


def make_ground_truth():
    volume, spacing = read_dicom("data/cxr/")
    bx, by, bz = np.array(volume.shape) * np.array(spacing) / 2
    true_params = torch.tensor([torch.pi, 0, torch.pi / 2, bx, by, bz])
    return volume, spacing, true_params


def get_perturbation():
    del_t = np.random.uniform(-torch.pi, torch.pi)
    del_p = np.random.uniform(0, 2 * torch.pi)
    del_g = np.random.uniform(-torch.pi / 2, 3 * torch.pi / 2)
    del_x = np.random.uniform(-10, 10)
    del_y = np.random.uniform(-10, 10)
    del_z = np.random.uniform(-10, 10)
    return torch.tensor([del_t, del_p, del_g, del_x, del_y, del_z])


def make_image(drr, true_params):
    perturbed_params = true_params + get_perturbation()
    perturbed = drr(200.0, *perturbed_params)
    return perturbed_params, perturbed


@click.command()
@click.option("--n_train", default=1000, help="Number of training DRRs to generate")
@click.option("--n_test", default=100, help="Number of test DRRs to generate")
@click.option("-o", "--outdir", default="dataset/", help="Output directory")
def main(n_train, n_test, outdir):
    volume, spacing, true_params = make_ground_truth()
    drr = DRR(volume, spacing, height=100, delx=5e-2, device="cuda")

    # Save the datasets
    outdir = Path(f"experiments/initialization/{outdir}")
    train = outdir / "train"
    test = outdir / "test"

    # Make the training set
    for i in tqdm(range(n_train)):
        outdir = train / str(i)
        outdir.mkdir(parents=True, exist_ok=True)
        perturbed_params, perturbed = make_image(drr, true_params)
        torch.save(perturbed, outdir / "image.pt")
        torch.save(perturbed_params, outdir / "params.pt")

    # Make the test set
    for i in tqdm(range(n_test)):
        outdir = test / str(i)
        outdir.mkdir(parents=True, exist_ok=True)
        perturbed_params, perturbed = make_image(drr, true_params)
        torch.save(perturbed, outdir / "image.pt")
        torch.save(perturbed_params, outdir / "params.pt")


if __name__ == "__main__":
    main()
