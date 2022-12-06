import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def plot_volume(volume, spacing, bx, by, bz, ax):

    # Get the dimensions of the volume
    Nx, Ny, Nz = np.array(volume.shape) * np.array(spacing)
    X, Y, Z = np.meshgrid(
        np.arange(0, Nx, spacing[0]) + bx,
        np.arange(0, Ny, spacing[1]) + by,
        -np.arange(0, Nz, spacing[2]) + bz,
    )

    kw = {
        "vmin": volume.min(),
        "vmax": volume.max(),
        "levels": np.linspace(volume.min(), volume.max(), 10),
        "cmap": "gray",
    }

    # Plot contour surfaces
    idx = 256
    _ = ax.contourf(
        X[:, :, 0], Y[:, :, 0], volume[:, :, 0], zdir="z", offset=Z.max(), **kw
    )
    _ = ax.contourf(
        X[idx, :, :], volume[idx, :, :], Z[idx, :, :], zdir="y", offset=Y.min(), **kw
    )
    _ = ax.contourf(
        volume[:, idx, :], Y[:, idx, :], Z[:, idx, :], zdir="x", offset=X.max(), **kw
    )

    # Set limits of the plot from coord limits
    cushion = 350
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(
        xlim=[xmin - cushion, xmax + cushion],
        ylim=[ymin - cushion, ymax + cushion],
        zlim=[zmin - cushion, zmax + cushion],
    )

    return ax


def plot_camera(source, rays, ax):

    # Plot the source
    source = source.detach().cpu().numpy()
    ax.scatter3D(source[0], source[1], source[2])

    # Plot the detector plane
    v0 = rays[0, 0, :].detach().cpu().numpy()
    v1 = rays[-1, 0, :].detach().cpu().numpy()
    v2 = rays[0, -1, :].detach().cpu().numpy()
    v3 = rays[-1, -1, :].detach().cpu().numpy()
    pts = np.array([v0, v1, v2, v3])
    ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2])
    verts = [[pts[0], pts[1], pts[3], pts[2]]]
    ax.add_collection3d(
        Poly3DCollection(
            verts,
            facecolors="cyan",
            linewidths=1,
            edgecolors="r",
            alpha=0.2,
        )
    )

    # Plot the rays
    nx, ny = rays.shape[:2]
    for i in range(0, nx, nx - 1):
        for j in range(0, ny, ny - 1):
            pt = rays[i, j, :].detach().cpu()
            ax.plot3D(
                [pt[0], source[0]],
                [pt[1], source[1]],
                [pt[2], source[2]],
                "k--",
                alpha=0.75,
            )

    return ax


def plot_drr(drr, title=None, ticks=True, animated=False, ax=None):
    """
    Plot an DRR output by the projector module.

    Inputs
    ------
    drr : torch.Tensor
        DRR image in torch tensor with shape (batch, height, width)
    title : str, optional
        Title for the plot
    ticks : Bool
        Toggle ticks and ticklabels.
    animated : Bool
        Set to true if using in animation.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axis to plot image on, if None is provided, a new axis will be made

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis with plotted image
    """
    drr = drr[0, :, :].detach().cpu()
    nx, ny = drr.shape
    if ax is None:
        _, ax = plt.subplots()
    img = ax.imshow(drr, cmap="gray", animated=animated)
    ax.xaxis.tick_top()
    ax.set(
        title=title,
        xticks=[0, nx - 1],
        xticklabels=[1, nx],
        yticks=[0, ny - 1],
        yticklabels=[1, ny],
    )
    if ticks is False:
        ax.set_xticks([])
        ax.set_yticks([])
    return img
