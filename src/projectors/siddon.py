import torch


class Siddon:
    """A vectorized version of the Siddon ray tracing algorithm."""

    def __init__(self, volume, spacing, device, eps=10e-10):
        self.volume = torch.tensor(volume, dtype=torch.float16, device=device)
        self.spacing = torch.tensor(spacing, dtype=torch.float32, device=device)
        self.device = device
        self.eps = eps

        # Reverse the rows to match the indexing scheme of the Siddon-Jacob's algorithm
        self.volume = torch.tensor(volume, dtype=torch.float16, device=device).flip([0])
        self.dims = torch.tensor(self.volume.shape, dtype=torch.float32, device=device)
        self.dims += 1.0

    def get_alpha_minmax(self, source, target):
        ssd = target - source + self.eps
        planes = torch.zeros(3, device=self.device)
        alpha0 = (planes * self.spacing - source) / ssd
        planes = self.dims - 1
        alpha1 = (planes * self.spacing - source) / ssd
        alphas = torch.stack([alpha0, alpha1])

        alphamin = alphas.min(dim=0).values.max(dim=-1).values
        alphamax = alphas.max(dim=0).values.min(dim=-1).values
        return alphamin, alphamax

    def get_alphas(self, source, target):
        # Get the CT sizing and spacing parameters
        dx, dy, dz = self.spacing
        nx, ny, nz = self.dims
        self.maxidx = ((nx - 1) * (ny - 1) * (nz - 1)).int().item() - 1

        # Get the alpha at each plane intersection
        sx, sy, sz = source
        alphax = torch.arange(nx, dtype=torch.float32, device=self.device) * dx - sx
        alphay = torch.arange(ny, dtype=torch.float32, device=self.device) * dy - sy
        alphaz = torch.arange(nz, dtype=torch.float32, device=self.device) * dz - sz

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
        # Drop indices where alphas for all rays are nan
        alphas = torch.sort(alphas, dim=0).values
        alphas = alphas[~alphas.isnan().all(dim=-1).all(dim=-1)]
        return alphas

    def get_voxel(self, alpha, source, target):
        sdd = target - source + self.eps  # source-to-detector distance
        idxs = ((source + alpha.unsqueeze(-1) * sdd) / self.spacing).trunc()
        idxs = (
            idxs[:, :, :, 0] * (self.dims[1] - 1) * (self.dims[2] - 1)
            + idxs[:, :, :, 1] * (self.dims[2] - 1)
            + idxs[:, :, :, 2]
        ).long() + 1

        # Conversion to long makes nan->-inf, so temporarily replace them with 0
        # This is cancelled out later by multiplication by nan step_length
        idxs[idxs < 0] = 0
        idxs[idxs > self.maxidx] = self.maxidx
        return torch.take(self.volume, idxs)

    def raytrace(self, source, target):
        alphas = self.get_alphas(source, target)
        alphamid = (alphas[0:-1] + alphas[1:]) / 2
        voxels = self.get_voxel(alphamid, source, target)

        # Step length for alphas out of range will be nan
        # These nans cancel out voxels convereted to 0 index
        step_length = torch.diff(alphas, dim=0)
        weighted_voxels = voxels * step_length

        drr = torch.nansum(weighted_voxels, dim=0)
        raylength = (target - source + self.eps).norm(dim=-1)
        drr *= raylength
        return drr
