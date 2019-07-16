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
import extendMSE


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

        self.conv_channels_up = nn.Conv2d(1, 32, 1)

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

        self.tconv_channels_down = nn.ConvTranspose2d(32, 1, 1)

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







import bmpReader
'''
argv:
1: 使用哪个显卡
'''

torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡
batchSize = int(sys.argv[2])
dReader = bmpReader.datasetReader(batchSize)


encNet = torch.load('./models/encNet_13_5.pkl', map_location='cuda:'+sys.argv[1]).cuda().eval()
decNet = torch.load('./models/decNet_13_5.pkl', map_location='cuda:'+sys.argv[1]).cuda().eval()


trainData = torch.empty([batchSize, 1, 256, 256]).float().cuda().requires_grad_(False)


'''
比较量化的ssim 以及获取数据分布使用的代码
for i in range(1, 256):

    for j in range(16): # 每16批 当作一个训练单元 统计这16批数据的表现
        for k in range(batchSize):
            trainData[k] = torch.from_numpy(dReader.readImg()).float().cuda()

        encData = encNet(trainData)
        qEncData = quantize(encData, i)
        #dataHistc = torch.histc(qEncData, min=0, max=i-1, bins=i)
        decData = decNet(qEncData / (i-1))

        currentMS_SSIM = pytorch_msssim.ms_ssim(trainData, decData, data_range=255, size_average=True)
        if(j==0):
            sumMS_SSIM = currentMS_SSIM.item()
            #sumDataHistc = dataHistc
        else:
            sumMS_SSIM = sumMS_SSIM + currentMS_SSIM.item()
            #sumDataHistc = sumDataHistc + dataHistc

    #print(i, (sumDataHistc.float()/sumDataHistc.sum()).detach().cpu().numpy())
    print(i, sumMS_SSIM/16)
'''


cList = []
for i in range(-1, 32):
    for j in range(16):  # 每16批 当作一个训练单元 统计这16批数据的表现
        for k in range(batchSize):
            trainData[k] = torch.from_numpy(dReader.readImg()).float().cuda()

        encData = encNet(trainData)
        if(i>=0):
            encData[:,i].zero_() # 设置i通道为0
        qEncData = quantize(encData, 256)
        decData = decNet(qEncData / 255)
        currentMS_SSIM = pytorch_msssim.ms_ssim(trainData, decData, data_range=255, size_average=True)

        if (j == 0):
            sumMS_SSIM = currentMS_SSIM.item()
        else:
            sumMS_SSIM = sumMS_SSIM + currentMS_SSIM.item()

    print(i, sumMS_SSIM / 16)
    cList.append((i, float(sumMS_SSIM) / 16))


cList.sort(key=lambda x:x[1])
for item in cList:
    print(item)









