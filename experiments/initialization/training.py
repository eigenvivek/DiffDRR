import click
import torch
from torch import optim
from torch.utils.data import DataLoader

from src import DRR, read_dicom

from dataloader import ParamDataset
from loss import GenLoss
from model import Model


def get_projector():
    volume, spacing = read_dicom("data/cxr/")
    return DRR(volume, spacing, height=100, delx=5e-2, device="cuda")


def get_model(lr):
    model = Model()
    drr = get_projector()
    loss_func = GenLoss(drr)
    opt = optim.SGD(model.parameters(), lr=lr)
    return model, loss_func, opt


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, val_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            _, _ = loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            loss, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in val_dl]
            )
        val_loss = sum(loss) / sum(nums)
        print(f"Epoch {epoch}: val_loss {val_loss:.4f}")


@click.command()
@click.option("--epochs", default=100, help="Number of epochs to train")
@click.option("--batch_size", default=32, help="Batch size")
@click.option("--lr", default=1e-3, help="Learning rate")
def main(epochs, batch_size, lr):
    # Get the training and testing datasets
    train_ds = ParamDataset("experiments/initialization/dataset/train")
    val_ds = ParamDataset("experiments/initialization/dataset/val")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    # Get the model and train
    model, loss_func, opt = get_model(lr)
    fit(epochs, model, loss_func, opt, train_dl, val_dl)


if __name__ == "__main__":
    main()
