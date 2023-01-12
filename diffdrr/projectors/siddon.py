import torch


class Siddon:
    """A vectorized version of the Siddon ray tracing algorithm."""

    def __init__(self, volume, spacing, dtype, device, eps=1e-9):
        self.spacing = torch.tensor(spacing, dtype=dtype, device=device)
        self.dtype = dtype
        self.device = device
        self.eps = eps

        # Reverse the rows to match the indexing scheme of the Siddon-Jacob's algorithm
        self.volume = torch.tensor(volume, dtype=self.dtype, device=self.device).flip([0])
        self.dims = torch.tensor(self.volume.shape, dtype=self.dtype, device=self.device)
        self.dims += 1.0

    def get_alpha_minmax(self, source, target):
        sdd = target - source + self.eps
        planes = torch.zeros(3, device=self.device)
        alpha0 = (planes * self.spacing - source) / sdd
        planes = self.dims - 1
        alpha1 = (planes * self.spacing - source) / sdd
        alphas = torch.stack([alpha0, alpha1])

        alphamin = alphas.min(dim=0).values.max(dim=-1).values.unsqueeze(-1)
        alphamax = alphas.max(dim=0).values.min(dim=-1).values.unsqueeze(-1)
        return alphamin, alphamax

    def get_alphas(self, source, target):
        # Get the CT sizing and spacing parameters
        dx, dy, dz = self.spacing
        nx, ny, nz = self.dims
        self.maxidx = ((nx - 1) * (ny - 1) * (nz - 1)).int().item() - 1

        # Get the alpha at each plane intersection
        sx, sy, sz = source[..., 0], source[..., 1], source[..., 2]
        alphax = torch.arange(nx, dtype=self.dtype, device=self.device) * dx
        alphay = torch.arange(ny, dtype=self.dtype, device=self.device) * dy
        alphaz = torch.arange(nz, dtype=self.dtype, device=self.device) * dz
        alphax = alphax.expand(len(source), -1) - sx.unsqueeze(-1)
        alphay = alphay.expand(len(source), -1) - sy.unsqueeze(-1)
        alphaz = alphaz.expand(len(source), -1) - sz.unsqueeze(-1)

        sdd = target - source + self.eps
        alphax = alphax / sdd[..., 0].unsqueeze(-1)
        alphay = alphay / sdd[..., 1].unsqueeze(-1)
        alphaz = alphaz / sdd[..., 2].unsqueeze(-1)
        alphas = torch.cat([alphax, alphay, alphaz], dim=-1)

        # Get the alphas within the range [alphamin, alphamax]
        alphamin, alphamax = self.get_alpha_minmax(source, target)
        good_idxs = torch.logical_and(alphas >= alphamin, alphas <= alphamax)
        alphas[~good_idxs] = torch.nan

        # Sort the alphas by ray, putting nans at the end of the list
        # Drop indices where alphas for all rays are nan
        alphas = torch.sort(alphas, dim=-1).values
        alphas = alphas[..., ~alphas.isnan().all(dim=0).all(dim=0)]
        return alphas

    def get_index(self, alpha, source, target):
        sdd = target - source + self.eps
        idxs = (source + alpha.unsqueeze(-1) * sdd.unsqueeze(2)) / self.spacing
        idxs = idxs.trunc()
        # Conversion to long makes nan->-inf, so temporarily replace them with 0
        # This is cancelled out later by multiplication by nan step_length
        idxs = (
            idxs[..., 0] * (self.dims[1] - 1) * (self.dims[2] - 1)
            + idxs[..., 1] * (self.dims[2] - 1)
            + idxs[..., 2]
        ).long() + 1
        idxs[idxs < 0] = 0
        idxs[idxs > self.maxidx] = self.maxidx
        return idxs

    def get_voxel(self, alpha, source, target):
        idxs = self.get_index(alpha, source, target)
        return torch.take(self.volume, idxs)

    def raytrace(self, source, target):
        alphas = self.get_alphas(source, target)
        alphamid = (alphas[..., 0:-1] + alphas[..., 1:]) / 2
        voxels = self.get_voxel(alphamid, source, target)

        # Step length for alphas out of range will be nan
        # These nans cancel out voxels convereted to 0 index
        step_length = torch.diff(alphas, dim=-1)
        weighted_voxels = voxels * step_length

        drr = torch.nansum(weighted_voxels, dim=-1)
        raylength = (target - source + self.eps).norm(dim=-1)
        drr *= raylength
        return drr
