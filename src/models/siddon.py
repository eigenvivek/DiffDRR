import torch


class Siddon:
    def __init__(self, spacing, center, volume):
        self.spacing = torch.tensor(spacing)
        self.center = torch.tensor(center)
        self.dims = torch.tensor(volume.shape) + 1.0
        self.volume = volume

    def get_alpha(self, planes, origin, target):
        return (self.center + planes * self.spacing - origin) / (target - origin)

    def get_alpha_minmax(self, origin, target):
        planes = torch.tensor([[1, 1, 1], self.dims]) - 1
        alphas = self.get_alpha(planes, origin, target)
        minis = torch.min(alphas, dim=0).values
        maxis = torch.max(alphas, dim=0).values
        alphamin = torch.max(minis)
        alphamax = torch.min(maxis)
        return alphamin, alphamax, minis, maxis

    def get_coords(self, alpha, origin, target):
        pxyz = origin + alpha * (target - origin)
        return (pxyz - self.center) / self.spacing

    def initialize(self, origin, target):
        alphamin, alphamax, minis, maxis = self.get_alpha_minmax(origin, target)
        idxmin = self.get_coords(alphamin, origin, target)
        idxmax = self.get_coords(alphamax, origin, target)

        # origin < target
        # get minidx
        a = (alphamin == minis) * torch.ones(3)
        b = (alphamin != minis) * (idxmin + 1).trunc()
        # get maxidx
        c = (alphamax == maxis) * (self.dims - 1)
        d = (alphamax != maxis) * idxmax.trunc()
        # origin > target
        # get minidx
        e = (alphamax == maxis) * torch.ones(3)
        f = (alphamax != maxis) * (idxmax + 1).trunc()
        # get maxidx
        g = (alphamin == minis) * (self.dims - 2)
        h = (alphamin != minis) * idxmin.trunc()

        minidx = (origin < target) * (a + b) + (origin >= target) * (e + f)
        maxidx = (origin < target) * (c + d) + (origin >= target) * (g + h)

        return alphamin, alphamax, minidx, maxidx

    def get_voxel_idx(self, alpha, origin, target):
        idxs = self.get_coords(alpha, origin, target).trunc().int()
        return list(idxs)

    def raytrace(self, origin, target):

        # Get the update conditions
        ones = torch.ones(3, dtype=int)
        update_idxs = (origin < target) * ones - (origin >= target) * ones
        update_alpha = self.spacing / torch.abs(target - origin)

        # Initialize the loop
        alphamin, alphamax, minidx, maxidx = self.initialize(origin, target)
        alphacurr = alphamin

        steps = self.get_alpha(
            minidx, origin, target
        )  # Get the potential next steps in the xyz planes
        idx = steps.argmin()  # Find the smallest step
        alphanext = steps[idx]  # I.e., the next plane

        alphamid = (alphacurr + alphanext) / 2
        voxel = self.get_voxel_idx(alphamid, origin, target)

        step_length = alphanext - alphacurr
        d12 = step_length * self.volume[voxel]
        alphacurr = alphanext.clone()

        # Loop over all voxels that the ray passes through
        while alphacurr < alphamax:
            voxel[idx] += update_idxs[idx]
            steps[idx] += update_alpha[idx]
            idx = steps.argmin()
            alphanext = steps[idx]
            step_length = alphanext - alphacurr
            d12 += step_length * self.volume[voxel]
            alphacurr = alphanext.clone()

        return d12
