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
    def forward(ctx, input):
        output = torch.round(input) # 量化
        return output

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output # 把量化器的导数当做1

def quantize(input):
    return Quantize.apply(input)


class EncodeNet(nn.Module):
    def __init__(self):
        super(EncodeNet, self).__init__()

        self.conv_channels_up = nn.Conv2d(1, 64, 1)

        self.conv_down_256_128 = nn.Conv2d(64, 64, 2, 2)
        self.conv_down_128_64 = nn.Conv2d(64, 64, 2, 2)
        self.conv_down_64_32 = nn.Conv2d(64, 64, 2, 2)
        self.conv_down_32_16 = nn.Conv2d(64, 64, 2, 2)

        self.gdn_down_256_128 = pytorch_gdn.GDN(64)
        self.gdn_down_128_64 = pytorch_gdn.GDN(64)
        self.gdn_down_64_32 = pytorch_gdn.GDN(64)
        self.gdn_down_32_16 = pytorch_gdn.GDN(64)

    def forward(self, x):

        x1 = F.leaky_relu(self.conv_channels_up(x))

        x2 = self.gdn_down_256_128(F.leaky_relu(self.conv_down_256_128(x1)))

        x3 = self.gdn_down_128_64(F.leaky_relu(self.conv_down_128_64(x2)))

        x4 = self.gdn_down_64_32(F.leaky_relu(self.conv_down_64_32(x3)))
        y4 = self.gdn_down_32_16(F.leaky_relu(self.conv_down_32_16(x4)))

        return y4







class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()

        self.tconv_channels_down = nn.ConvTranspose2d(64, 1, 1)

        self.tconv_up_16_32 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.tconv_up_32_64 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.tconv_up_64_128 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.tconv_up_128_256 = nn.ConvTranspose2d(64, 64, 2, 2)

        self.igdn_up_16_32 = pytorch_gdn.GDN(64, True)
        self.igdn_up_32_64 = pytorch_gdn.GDN(64, True)
        self.igdn_up_64_128 = pytorch_gdn.GDN(64, True)
        self.igdn_up_128_256 = pytorch_gdn.GDN(64, True)

    def forward(self, x):
        x4 = F.leaky_relu(self.tconv_up_16_32(self.igdn_up_16_32(x)))

        x3 = F.leaky_relu(self.tconv_up_32_64(self.igdn_up_32_64(x4)))

        x2 = F.leaky_relu(self.tconv_up_64_128(self.igdn_up_64_128(x3)))

        x1 = F.leaky_relu(self.tconv_up_128_256(self.igdn_up_128_256(x2)))

        x = F.leaky_relu(self.tconv_channels_down(x1))

        return x








torch.cuda.set_device(0) # 设置使用哪个显卡

encNet1 = torch.load('./models/encNet1_10_1.pkl', map_location='cuda:0').cuda()
decNet1 = torch.load('./models/decNet1_10_1.pkl', map_location='cuda:0').cuda()
encNet2 = torch.load('./models/encNet2_10_1.pkl', map_location='cuda:0').cuda()
decNet2 = torch.load('./models/decNet2_10_1.pkl', map_location='cuda:0').cuda()

MSELoss = nn.MSELoss()


img = Image.open('./test.bmp').convert('L')
inputData = torch.from_numpy(numpy.asarray(img).astype(float).reshape([1, 1, 256, 256])).float().cuda()

encData1 = encNet1(inputData)
qEncData1 = quantize(encData1)
decData1 = decNet1(qEncData1)
encData2 = encNet2(decData1)
qEncData2 = quantize(encData2)
decData2 = decNet2(qEncData2)
decData = decData1 + decData2

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

qEncData = qEncData1.detach().cpu().numpy().astype(int)
print(qEncData.min(),qEncData.max(),qEncData.shape)
numpy.save('./output/MS-SSIM'+str(MS_SSIM.item()),qEncData)

'''
img1.save('./test.jpg',quality=8)
img2 = Image.open('./test.jpg')
img1 = torch.from_numpy(numpy.asarray(img1).astype(float).reshape([1, 1, 256, 256])).float().cuda()
img2 = torch.from_numpy(numpy.asarray(img2).astype(float).reshape([1, 1, 256, 256])).float().cuda()
MS_SSIM = pytorch_msssim.ms_ssim(img1, img2, data_range=255, size_average=True)
print(MS_SSIM)
'''









