import torch.nn as nn

from src.metrics import XCorr2


class GenLoss(nn.Module):
    def __init__(self, drr, sdr=200.0, zero_mean_normalized=True):
        super(GenLoss, self).__init__()
        self.drr = drr
        self.sdr = sdr
        self.xcorr2 = XCorr2(zero_mean_normalized)

    def forward(self, est_params, true_params):
        est = self.drr(self.sdr, *est_params.detach().cpu().tolist()[0])
        true = self.drr(self.sdr, *true_params.detach().cpu().tolist()[0])
        loss = self.xcorr2(est, true)
        return loss
