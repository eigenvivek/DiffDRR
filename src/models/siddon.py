import torch


class Siddon:
    """A vectorized version of the Siddon ray tracing algorithm."""

    def __init__(self, spacing, isocenter, volume, device, eps=10e-10):
        self.spacing = torch.tensor(spacing, dtype=torch.float32, device=device)
        self.isocenter = torch.tensor(isocenter, dtype=torch.float32, device=device)
        self.dims = torch.tensor(volume.shape, dtype=torch.float32, device=device)
        self.volume = torch.tensor(volume, dtype=torch.float16, device=device)
        self.device = device
        self.eps = eps

    def get_alpha_minmax(self, source, target):
        ssd = target - source + self.eps
        planes = torch.tensor([0, 0, 0], device=self.device)
        alpha0 = (self.isocenter + planes * self.spacing - source) / ssd
        alpha1 = (self.isocenter + self.dims * self.spacing - source) / ssd
        alphas = torch.stack([alpha0, alpha1])

        minis = torch.min(alphas, dim=0).values
        maxis = torch.max(alphas, dim=0).values
        alphamin = torch.max(minis, dim=-1).values
        alphamax = torch.min(maxis, dim=-1).values
        return alphamin, alphamax

    def get_alphas(self, source, target):
        # Get the CT sizing and spacing parameters
        nx, ny, nz = self.volume.shape
        dx, dy, dz = self.spacing

        # Get the alpha at each plane intersection
        sx, sy, sz = source
        alphax = torch.arange(nx, dtype=torch.float32, device=self.device) * dx + sx
        alphay = torch.arange(ny, dtype=torch.float32, device=self.device) * dy + sy
        alphaz = torch.arange(nz, dtype=torch.float32, device=self.device) * dz + sz

        sdd = target - source + self.eps  # source-to-detector distance
        alphax = alphax.unsqueeze(-1).unsqueeze(-1) / sdd[:, :, 0]
        alphay = alphay.unsqueeze(-1).unsqueeze(-1) / sdd[:, :, 1]
        alphaz = alphaz.unsqueeze(-1).unsqueeze(-1) / sdd[:, :, 2]
        alphas = torch.vstack([alphax, alphay, alphaz])

        # Get the alphas within the range [alphamin, alphamax]
        alphamin, alphamax = self.get_alpha_minmax(source, target)
        good_idxs = torch.logical_and(alphas >= alphamin, alphas <= alphamax)
        alphas[~good_idxs] = torch.nan

        # Sort the alphas by ray, putting nans at the end of the list
        alphas = torch.sort(alphas, dim=0).values
        return alphas

    def get_voxel(self, alpha, source, target):
        sdd = target - source + self.eps  # source-to-detector distance
        idxs = (
            (source + alpha.unsqueeze(-1) * sdd - self.isocenter) / self.spacing
        ).trunc()
        idxs = (
            idxs[:, :, :, 0]
            + idxs[:, :, :, 1] * self.volume.shape[1]
            + idxs[:, :, :, 2] * self.volume.shape[2]
        ).long()

        # Conversion to long makes nan->-inf, so to replace them with 0 temporarily
        # This is cancelled out later by multiplication by nan step_length
        idxs[idxs < 0] = 0
        return torch.take(self.volume, idxs)

    def raytrace(self, source, target):
        alphas = self.get_alphas(source, target)
        alphamid = (alphas[0:-1] + alphas[1:]) / 2
        voxels = self.get_voxel(alphamid, source, target)

        # Step length for alphas out of range will be nan
        # These nans cancel out voxels convereted to 0
        step_length = torch.diff(alphas, dim=0)
        weighted_voxels = voxels * step_length
        return torch.nansum(weighted_voxels, dim=0)
