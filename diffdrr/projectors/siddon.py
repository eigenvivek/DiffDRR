import torch

EPS = 1e-9


def siddon_raycast(source, target, volume, spacing):
    dims = torch.tensor(volume.shape) + 1
    alphas, maxidx = get_alphas(source, target, spacing, dims)
    alphamid = (alphas[..., 0:-1] + alphas[..., 1:]) / 2
    voxels = get_voxel(alphamid, source, target, volume, spacing, dims, maxidx)

    # Step length for alphas out of range will be nan
    # These nans cancel out voxels convereted to 0 index
    step_length = torch.diff(alphas, dim=-1)
    weighted_voxels = voxels * step_length

    drr = torch.nansum(weighted_voxels, dim=-1)
    raylength = (target - source + EPS).norm(dim=-1)
    drr *= raylength
    return drr


def get_alphas(source, target, spacing, dims):
    # Get the CT sizing and spacing parameters
    dx, dy, dz = spacing
    nx, ny, nz = dims
    maxidx = ((nx - 1) * (ny - 1) * (nz - 1)).int().item() - 1

    # Get the alpha at each plane intersection
    sx, sy, sz = source[..., 0], source[..., 1], source[..., 2]
    alphax = torch.arange(nx).to(source) * dx
    alphay = torch.arange(ny).to(source) * dy
    alphaz = torch.arange(nz).to(source) * dz
    alphax = alphax.expand(len(source), 1, -1) - sx.unsqueeze(-1)
    alphay = alphay.expand(len(source), 1, -1) - sy.unsqueeze(-1)
    alphaz = alphaz.expand(len(source), 1, -1) - sz.unsqueeze(-1)

    sdd = target - source + EPS
    alphax = alphax / sdd[..., 0].unsqueeze(-1)
    alphay = alphay / sdd[..., 1].unsqueeze(-1)
    alphaz = alphaz / sdd[..., 2].unsqueeze(-1)
    alphas = torch.cat([alphax, alphay, alphaz], dim=-1)

    # # Get the alphas within the range [alphamin, alphamax]
    alphamin, alphamax = get_alpha_minmax(source, target, spacing, dims)
    good_idxs = torch.logical_and(alphas >= alphamin, alphas <= alphamax)
    alphas[~good_idxs] = torch.nan

    # # Sort the alphas by ray, putting nans at the end of the list
    # # Drop indices where alphas for all rays are nan
    alphas = torch.sort(alphas, dim=-1).values
    alphas = alphas[..., ~alphas.isnan().all(dim=0).all(dim=0)]
    return alphas, maxidx


def get_alpha_minmax(source, target, spacing, dims):
    sdd = target - source + EPS
    planes = torch.zeros(3).to(source)
    alpha0 = (planes * spacing - source) / sdd
    planes = (dims - 1).to(source)
    alpha1 = (planes * spacing - source) / sdd
    alphas = torch.stack([alpha0, alpha1]).to(source)

    alphamin = alphas.min(dim=0).values.max(dim=-1).values.unsqueeze(-1)
    alphamax = alphas.max(dim=0).values.min(dim=-1).values.unsqueeze(-1)
    return alphamin, alphamax


def get_voxel(alpha, source, target, volume, spacing, dims, maxidx):
    idxs = get_index(alpha, source, target, spacing, dims, maxidx)
    return torch.take(volume, idxs)


def get_index(alpha, source, target, spacing, dims, maxidx):
    sdd = target - source + EPS
    idxs = source.unsqueeze(1) + alpha.unsqueeze(-1) * sdd.unsqueeze(2)
    idxs = idxs / spacing
    idxs = idxs.trunc()
    # Conversion to long makes nan->-inf, so temporarily replace them with 0
    # This is cancelled out later by multiplication by nan step_length
    idxs = (
        idxs[..., 0] * (dims[1] - 1) * (dims[2] - 1)
        + idxs[..., 1] * (dims[2] - 1)
        + idxs[..., 2]
    ).long() + 1
    idxs[idxs < 0] = 0
    idxs[idxs > maxidx] = maxidx
    return idxs
