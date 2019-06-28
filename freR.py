import torch
import torch.nn.functional as F
class freR(torch.nn.Module):
    def __init__(self):
        super(freR, self).__init__()

    def forward(ctx, x, y):
        xf = torch.rfft(x, 3, True)
        yf = torch.rfft(y, 3, True)
        R = (xf*yf)/(xf.norm()*yf.norm())
        return R.sum().acos()


freRLoss = freR()
