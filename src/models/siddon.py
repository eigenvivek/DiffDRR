import torch


class Siddon:
    def __init__(self, spacing, isocenter, volume, device, eps=10e-10):
        self.spacing = torch.tensor(spacing, dtype=torch.float32, device=device)
        self.isocenter = torch.tensor(isocenter, dtype=torch.float32, device=device)
        self.dims = torch.tensor(volume.shape, dtype=torch.float32, device=device) + 1.0
        self.volume = torch.tensor(volume, dtype=torch.float16, device=device)
        self.device = device
        self.eps = eps

    def get_alpha(self, planes, source, target):
        return (self.isocenter + planes * self.spacing - source) / (
            target - source + self.eps
        )

    def get_alpha_minmax(self, source, target):
        planes = torch.tensor([0, 0, 0], device=self.device)
        alpha0 = self.get_alpha(planes, source, target)
        planes = self.dims - 1
        alpha1 = self.get_alpha(planes, source, target)
        alphas = torch.stack([alpha0, alpha1])

        minis = torch.min(alphas, dim=0).values
        maxis = torch.max(alphas, dim=0).values
        alphamin = torch.max(minis, dim=-1).values
        alphamax = torch.min(maxis, dim=-1).values
        return alphamin, alphamax, minis, maxis

    def get_coords(self, alpha, source, target):
        pxyz = source + alpha.unsqueeze(-1) * (target - source + self.eps)
        return (pxyz - self.isocenter) / self.spacing

    def initialize(self, source, target):
        alphamin, alphamax, minis, maxis = self.get_alpha_minmax(source, target)
        alphamin_ = alphamin.unsqueeze(-1)
        alphamax_ = alphamax.unsqueeze(-1)

        idxmin = self.get_coords(alphamin, source, target)
        idxmax = self.get_coords(alphamax, source, target)

        # source < target
        # get minidx
        a = (alphamin_ == minis) * torch.ones(3, device=self.device)
        b = (alphamin_ != minis) * (idxmin + 1).trunc()
        # get maxidx
        c = (alphamax_ == maxis) * (self.dims - 1)
        d = (alphamax_ != maxis) * idxmax.trunc()
        # source > target
        # get minidx
        e = (alphamax_ == maxis) * torch.ones(3, device=self.device)
        f = (alphamax_ != maxis) * (idxmax + 1).trunc()
        # get maxidx
        g = (alphamin_ == minis) * (self.dims - 2)
        h = (alphamin_ != minis) * idxmin.trunc()

        minidx = (source < target) * (a + b) + (source >= target) * (e + f)
        maxidx = (source < target) * (c + d) + (source >= target) * (g + h)
        n_iters = (maxidx - minidx).max(dim=0).values.max(
            dim=0
        ).values.sum().int().item() + 1

        return alphamin, alphamax, minidx, maxidx, n_iters

    def get_next_step(self, steps):
        alphanext, idxs = steps.min(dim=-1)
        idxs = steps.argmin(dim=-1)
        idxs = torch.dstack([idxs == 0, idxs == 1, idxs == 2])
        return alphanext, idxs

    def get_voxel_idx(self, alpha, source, target):
        idxs = self.get_coords(alpha, source, target).trunc()
        return idxs

    def get_voxel(self, idxs):
        idxs = (
            idxs[:, :, 0]
            + idxs[:, :, 1] * self.volume.shape[1]
            + idxs[:, :, 2] * self.volume.shape[2]
        )
        return torch.take(self.volume, idxs.long())

    def get_update(self, alphacurr, alphamax):
        update_1 = alphacurr < alphamax
        update_2 = torch.isclose(alphacurr, alphamax)
        update_2 = torch.logical_not(update_2)
        update = torch.logical_and(update_1, update_2)
        return update.unsqueeze(-1)

    def raytrace(self, source, target):

        # Get the update conditions
        ones = torch.ones(3, device=self.device)
        update_idxs = (source < target) * ones - (source > target) * ones
        update_alpha = self.spacing / torch.abs(target - source + self.eps)

        # Initialize the loop
        alphamin, alphamax, minidx, _, n_iters = self.initialize(source, target)
        alphamax = alphamax.clone()
        alphacurr = alphamin.clone()

        # Get the potential next steps in the xyz planes
        steps = self.get_alpha(minidx, source, target)
        alphanext, idxs = self.get_next_step(steps)

        alphamids = (alphacurr + alphanext) / 2
        voxelidxs = self.get_voxel_idx(alphamids, source, target)

        drr = (alphanext - alphacurr) * self.get_voxel(voxelidxs)
        alphacurr = alphanext.clone()

        # Loop over all voxels that the ray passes through
        for _ in range(n_iters):
            update = self.get_update(alphacurr, alphamax)
            voxelidxs += update_idxs * idxs * update
            steps += update_alpha * idxs * update
            alphanext, idxs = self.get_next_step(steps)
            drr += (alphanext - alphacurr) * self.get_voxel(voxelidxs)
            alphacurr = alphanext.clone()

        return drr
