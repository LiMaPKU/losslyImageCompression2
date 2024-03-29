from PIL import Image
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim
import sys
import os
import pytorch_gdn
import pytorch_msssim
class Quantize(torch.autograd.Function): # 量化函数
    @staticmethod
    def forward(ctx, input, qLevel):
        # input的范围需要在[0,1]
        # qLevel是量化等级 例如qLevel=4 input变换到[0,3]然后就近量化
        input = input * (qLevel - 1)
        output = torch.round(input) # 量化
        return output

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None # 把量化器的导数当做1

def quantize(input, qLevel):
    return Quantize.apply(input, qLevel)

def vecNorm(x):
    return (x - x.min()) / (x.max() - x.min())

class EncodeNet(nn.Module):
    def __init__(self):
        super(EncodeNet, self).__init__()

        self.conv_channels_up = nn.Conv2d(1, 32, 5, padding=2)

        self.conv_down_256_128 = nn.Conv2d(32, 32, 2, 2)

        self.conv_down_128_64 = nn.Conv2d(32, 32, 2, 2)

        self.conv_down_64_32 = nn.Conv2d(32, 32, 2, 2)

        self.conv_down_32_16 = nn.Conv2d(32, 32, 2, 2)

        self.gdn_down_256_128 = pytorch_gdn.GDN(32)
        self.gdn_down_128_64 = pytorch_gdn.GDN(32)
        self.gdn_down_64_32 = pytorch_gdn.GDN(32)
        self.gdn_down_32_16 = pytorch_gdn.GDN(32)

    def forward(self, x):

        x = x / 255

        x1 = F.leaky_relu(self.conv_channels_up(x))

        x2 = self.gdn_down_256_128(F.leaky_relu(self.conv_down_256_128(x1)))

        x3 = self.gdn_down_128_64(F.leaky_relu(self.conv_down_128_64(x2)))

        x4 = self.gdn_down_64_32(F.leaky_relu(self.conv_down_64_32(x3)))

        x5 = self.gdn_down_32_16(F.leaky_relu(self.conv_down_32_16(x4)))

        return vecNorm(x5)







class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()

        self.tconv_channels_down = nn.ConvTranspose2d(32, 1, 5, padding=2)

        self.tconv_up_16_32 = nn.ConvTranspose2d(32, 32, 2, 2)

        self.tconv_up_32_64 = nn.ConvTranspose2d(32, 32, 2, 2)

        self.tconv_up_64_128 = nn.ConvTranspose2d(32, 32, 2, 2)

        self.tconv_up_128_256 = nn.ConvTranspose2d(32, 32, 2, 2)


        self.igdn_up_16_32 = pytorch_gdn.GDN(32, True)
        self.igdn_up_32_64 = pytorch_gdn.GDN(32, True)
        self.igdn_up_64_128 = pytorch_gdn.GDN(32, True)
        self.igdn_up_128_256 = pytorch_gdn.GDN(32, True)

    def forward(self, x5):

        x4 = F.leaky_relu(self.tconv_up_16_32(self.igdn_up_16_32(x5)))

        x3 = F.leaky_relu(self.tconv_up_32_64(self.igdn_up_32_64(x4)))

        x2 = F.leaky_relu(self.tconv_up_64_128(self.igdn_up_64_128(x3)))

        x1 = F.leaky_relu(self.tconv_up_128_256(self.igdn_up_128_256(x2)))

        x = F.leaky_relu(self.tconv_channels_down(x1))

        x = x * 255

        return x





torch.cuda.set_device(0) # 设置使用哪个显卡

encNet = torch.load('./models/encNet_16.pkl', map_location='cuda:0').cuda()
decNet = torch.load('./models/decNet_16.pkl', map_location='cuda:0').cuda()

MSELoss = nn.MSELoss()


img = Image.open('./test.bmp').convert('L')
inputData = torch.from_numpy(numpy.asarray(img).astype(float).reshape([1, 1, 256, 256])).float().cuda()

encData = encNet(inputData)
qEncData = quantize(encData, 4)
decData = decNet(qEncData / 3)


MSEL = MSELoss(inputData, decData)
img1 = inputData.clone()
img2 = decData.clone()
img1.detach_()
img2.detach_()
img2[img2<0] = 0
img2[img2>255] = 255
MS_SSIM = pytorch_msssim.ms_ssim(img1, img2, data_range=255, size_average=True)
print('MSEL=','%.3f'%MSEL.item(), 'MS_SSIM=','%.3f'%MS_SSIM)
img2 = img2.cpu().numpy().astype(int).reshape([256, 256])
img2 = Image.fromarray(img2.astype('uint8')).convert('L')
img2.save('./output/output.bmp')
img1 = img1.cpu().numpy().astype(int).reshape([256, 256])
img1 = Image.fromarray(img1.astype('uint8')).convert('L')
img1.save('./output/input.bmp')

numpy.save('./output/output.npy',qEncData.detach().cpu().numpy().astype(int))








