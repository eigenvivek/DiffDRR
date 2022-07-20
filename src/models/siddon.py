import torch


class Siddon:
    def __init__(self, spacing, isocenter, volume, device=device):
        self.spacing = torch.tensor(spacing, device=device)
        self.isocenter = torch.tensor(isocenter, device=device)
        self.dims = torch.tensor(volume.shape, device=device) + 1.0
        self.volume = torch.tensor(volume, device=device)
        self.device = device

    def get_alpha(self, planes, source, target):
        return (self.isocenter + planes * self.spacing - source) / (target - source)

    def get_alpha_minmax(self, source, target):
        planes = torch.tensor([0, 0, 0], device=self.device)
        alpha0 = (self.isocenter + planes * self.spacing - source) / (target - source)
        planes = self.dims - 1
        alpha1 = (self.isocenter + planes * self.spacing - source) / (target - source)
        alphas = torch.stack([alpha0, alpha1])

        minis = torch.min(alphas, dim=0).values
        maxis = torch.max(alphas, dim=0).values
        alphamin = torch.max(minis, dim=-1).values
        alphamax = torch.min(maxis, dim=-1).values
        return alphamin, alphamax, minis, maxis

    def get_coords(self, alpha, source, target):
        pxyz = source + alpha * (target - source)
        return (pxyz - self.isocenter) / self.spacing

    def initialize(self, source, target):
        alphamin, alphamax, minis, maxis = self.get_alpha_minmax(source, target)
        alphamin = alphamin.expand(3, -1, -1).permute(1, 2, 0)
        alphamax = alphamax.expand(3, -1, -1).permute(1, 2, 0)
        idxmin = self.get_coords(alphamin, source, target)
        idxmax = self.get_coords(alphamax, source, target)

        # source < target
        # get minidx
        a = (alphamin == minis) * torch.ones(3, device=self.device)
        b = (alphamin != minis) * (idxmin + 1).trunc()
        # get maxidx
        c = (alphamax == maxis) * (self.dims - 1)
        d = (alphamax != maxis) * idxmax.trunc()
        # source > target
        # get minidx
        e = (alphamax == maxis) * torch.ones(3, device=self.device)
        f = (alphamax != maxis) * (idxmax + 1).trunc()
        # get maxidx
        g = (alphamin == minis) * (self.dims - 2)
        h = (alphamin != minis) * idxmin.trunc()

        minidx = (source < target) * (a + b) + (source >= target) * (e + f)
        maxidx = (source < target) * (c + d) + (source >= target) * (g + h)

        return alphamin, alphamax, minidx, maxidx

    def get_voxel_idx(self, alpha, source, target):
        idxs = self.get_coords(alpha.expand(3, -1, -1).permute(1, 2, 0), source, target)
        idxs = idxs.trunc().long()
        idxs = (
            idxs[:, :, 0]
            + idxs[:, :, 1] * self.volume.shape[1]
            + idxs[:, :, 2] * self.volume.shape[2]
        )
        return idxs

    def get_voxel(self, voxel_idxs):
        return torch.take(self.volume, voxel_idxs)

    def raytrace(self, source, target):

        # Get the update conditions
        ones = torch.ones(3, device=self.device, dtype=int)
        update_idxs = (source < target) * ones - (source >= target) * ones
        update_alpha = self.spacing / torch.abs(target - source)

        # Initialize the loop
        alphamin, alphamax, minidx, maxidx = self.initialize(source, target)
        alphacurr = alphamin[:, :, 0].clone()

        # Get the potential next steps in the xyz planes
        steps = self.get_alpha(minidx, source, target)
        idx = steps.argmin(dim=-1)  # Find the smallest step
        alphanext = steps[idx]  # I.e., the next plane

        alphamid = (alphacurr + alphanext) / 2
        voxel = self.get_voxel_idx(alphamid, source, target)

        step_length = alphanext - alphacurr
        d12 = step_length * self.volume[voxel]
        alphacurr = alphanext.clone()

        # Loop over all voxels that the ray passes through
        while alphacurr < alphamax and not torch.isclose(alphacurr, alphamax):
            voxel[idx] += update_idxs[idx]
            steps[idx] += update_alpha[idx]
            idx = steps.argmin()
            alphanext = steps[idx]
            step_length = alphanext - alphacurr
            d12 += step_length * self.volume[voxel]
            alphacurr = alphanext.clone()

        return d12
