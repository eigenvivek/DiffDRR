import tempfile

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .visualize import plot_drr


def animate(out, df, sdr, drr, ground_truth=None, verbose=True):
    """Animate the optimization of a DRR."""
    # Make the axes
    if ground_truth is None:

        def make_fig():
            fig, ax_opt = plt.subplots(
                figsize=(3, 3),
                constrained_layout=True,
            )
            return fig, ax_opt

    else:

        def make_fig(ground_truth):
            fig, (ax_fix, ax_opt) = plt.subplots(
                ncols=2,
                figsize=(6, 3),
                constrained_layout=True,
            )
            plot_drr(ground_truth, ax=ax_fix)
            ax_fix.set(xlabel="Fixed DRR")
            return fig, ax_opt

    # Compute DRRs, plot, and save to temporary folder
    if verbose:
        itr = tqdm(df.iterrows(), desc="Precomputing DRRs", total=len(df))
    else:
        itr = df.iterrows()

    with tempfile.TemporaryDirectory() as tmpdir:
        idxs = []
        for idx, row in itr:
            fig, ax_opt = make_fig(ground_truth)
            params = row[["theta", "phi", "gamma", "bx", "by", "bz"]].values
            itr = drr(sdr, *params)
            _ = plot_drr(itr, ax=ax_opt)
            ax_opt.set(xlabel="Moving DRR")
            fig.savefig(f"{tmpdir}/{idx}.png")
            plt.close(fig)
            idxs.append(idx)
        frames = np.stack(
            [iio.imread(f"{tmpdir}/{idx}.png") for idx in idxs],
            axis=0,
        )

    # Make the animation
    iio.imwrite(out, frames)
