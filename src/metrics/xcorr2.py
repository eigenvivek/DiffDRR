# Adapted from: https://github.com/connorlee77/pytorch-xcorr2/blob/master/xcorr2.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class xcorr2(nn.Module):

    """
    Correlates two images. Images must be of the same size.
    """

    def __init__(self, zero_mean_normalize=True):
        super(xcorr2, self).__init__()
        self.InstanceNorm = nn.InstanceNorm2d(
            1, affine=False, track_running_stats=False
        )
        self.zero_mean_normalize = zero_mean_normalize

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): Batch of Img1 of dimensions [B, C, H, W].
            x2 (torch.Tensor): Batch of Img2 of dimensions [B, C, H, W].
        Returns:
            scores (torch.Tensor): The correlation scores for the pairs. The output shape is [8, 1].
        """

        if x1.shape == x2.shape:
            scores = self.match_corr_same_size(x1, x2)
        else:
            scores = self.match_corr(x1, x2)

        return scores

    def match_corr_same_size(self, x1, x2):
        b, c, h, w = x2.shape

        if self.zero_mean_normalize:
            x1 = self.InstanceNorm(x1)
            x2 = self.InstanceNorm(x2)

        scores = torch.matmul(x1.view(b, 1, c * h * w), x2.view(b, c * h * w, 1))
        scores /= h * w * c

        return scores

    def match_corr(self, embed_ref, embed_srch):
        """ Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].
        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.
        Returns:
            match_map: (torch.Tensor) The correlation between
        """
        b, c, h, w = embed_srch.shape

        # Here the correlation layer is implemented using a trick with the
        # conv2d function using groups in order to do the correlation with
        # batch dimension. Basically we concatenate each element of the batch
        # in the channel dimension for the search image (making it
        # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
        # the batch. This grouped convolution/correlation is equivalent to a
        # correlation between the two images, though it is not obvious.
        if self.zero_mean_normalize:
            embed_ref = self.InstanceNorm(embed_ref)
            embed_srch = self.InstanceNorm(embed_srch)

        # Has problems with mixed-precision training
        match_map = F.conv2d(
            embed_srch.view(1, b * c, h, w), embed_ref, groups=b
        ).float()
        match_map /= self.img_size * self.img_size * c

        # Here we reorder the dimensions to get back the batch dimension.
        match_map = match_map.permute(1, 0, 2, 3)

        return match_map
