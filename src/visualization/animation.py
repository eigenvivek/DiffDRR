import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from tqdm import tqdm

from .visualize import plot_drr


def _precompute_drrs(df, sdr, drr, ax):
    """Precompute the DRRs and save them to an artist."""
    imgs = []
    for _, row in tqdm(df.iterrows(), desc="Precomputing DRRs", total=len(df)):
        params = row[["theta", "phi", "gamma", "bx", "by", "bz"]].values
        itr = drr(200, *params)
        img = plot_drr(itr, animated=True, ax=ax)
        imgs.append([img])
    return imgs


def animate(df, sdr, drr, ground_truth=None):
    """Animate the optimization of a DRR."""
    if ground_truth is None:
        fig, ax_opt = plt.subplots(dpi=300)
    else:
        fig, (ax_opt, ax_fix) = plt.subplots(ncols=2, dpi=300)

    imgs = _precompute_drrs(df, sdr, drr, ax=ax_opt)
    ax_opt.set(xlabel="Moving DRR")

    if ground_truth is not None:
        plot_drr(ground_truth, ax=ax_fix)
        ax_fix.set(xlabel="Fixed DRR")
        
    anim = ArtistAnimation(fig, imgs, interval=50, blit=True, repeat_delay=1000)
    return anim.to_jshtml()
