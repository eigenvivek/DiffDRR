import torch


def reshape_subsampled_drr(img, detector, batch_size):
    n_points = detector.height * detector.width
    drr = torch.zeros(batch_size, n_points).to(detector.device)
    drr[:, detector.subsamples[-1]] = img
    drr = drr.view(batch_size, detector.height, detector.width)
    return drr
