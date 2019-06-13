import torch
import torch.nn.functional as F

exp_g = 2.6
exp_s = 2
exp_f = 0.6


class nlpDistanceLoss(torch.nn.Module):
    def __init__(self):
            super(nlpDistanceLoss, self).__init__()

    def forward(ctx, ima, imb, lpLevel): # lpLelel 拉普拉斯金字塔层数
        # im: b*c*m*n 一般要求m==n
        lowPassFilter = torch.tensor([
            [0.0025, 0.0125, 0.02, 0.0125, 0.0025],
            [0.0125, 0.0625, 0.10, 0.0625, 0.0125],
            [0.0200, 0.1000, 0.16, 0.1000, 0.0200],
            [0.0125, 0.0625, 0.10, 0.0625, 0.0125],
            [0.0025, 0.0125, 0.02, 0.0125, 0.0025]
        ]).float().unsqueeze(0).unsqueeze(0).cuda()

        pFilter = torch.tensor([
            [0.04, 0.04, 0.05, 0.04, 0.04],
            [0.04, 0.03, 0.04, 0.03, 0.04],
            [0.05, 0.04, 0.05, 0.04, 0.05],
            [0.04, 0.03, 0.04, 0.03, 0.04],
            [0.04, 0.04, 0.05, 0.04, 0.04]
        ]).float().unsqueeze(0).unsqueeze(0).cuda()

        xak = ima
        xbk = imb
        ret = torch.zeros([1]).cuda()

        for i in range(lpLevel):
            Ns = torch.tensor([xak.shape[2] * ima.shape[3]]).float().cuda()
            xak_ = F.interpolate(F.conv2d(xak, lowPassFilter, padding=2), scale_factor=[0.5,0.5], mode='bilinear', align_corners=False)
            zak = xak - F.conv2d(F.interpolate(xak_, scale_factor=[2,2], mode='bilinear', align_corners=False), lowPassFilter, padding=2)
            fak = F.conv2d(torch.abs(zak), pFilter, padding=2)
            fak = fak + fak.mean(2).mean(2).unsqueeze(2).unsqueeze(3)
            yak = zak / fak

            xbk_ = F.interpolate(F.conv2d(xbk, lowPassFilter, padding=2), scale_factor=[0.5, 0.5], mode='bilinear',align_corners=False)
            zbk = xbk - F.conv2d(F.interpolate(xak_, scale_factor=[2, 2], mode='bilinear', align_corners=False), lowPassFilter, padding=2)
            fbk = F.conv2d(torch.abs(zbk), pFilter, padding=2)
            fbk = fbk + fbk.mean(2).mean(2).unsqueeze(2).unsqueeze(3)
            ybk = zbk / fbk

            ret = ret + torch.pow(yak - ybk, 2).sum()/torch.sqrt(Ns)

            xak = xak_
            xbk = xbk_

        return ret / lpLevel / ima.shape[0]

NLPLoss = nlpDistanceLoss()

if __name__ == '__main__':  # 如果运行本py文件 就运行main函数
    from PIL import Image
    import numpy

    img1 = Image.open('./0.bmp').convert('L')
    img2 = Image.open('./1.bmp').convert('L')
    img1 = numpy.asarray(img1).astype(float).reshape([1, 256, 256])
    img1 = torch.from_numpy(img1).unsqueeze(0).float().cuda()
    img1.requires_grad_(True)
    img2 = numpy.asarray(img2).astype(float).reshape([1, 256, 256])
    img2 = torch.from_numpy(img2).unsqueeze(0).float().cuda()
    img2.requires_grad_(True)
    y = NLPLoss(img1, img2, 5)
    print(y)
    y.backward()


