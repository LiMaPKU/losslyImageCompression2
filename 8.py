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

# 导入信息熵损失
from torch.utils.cpp_extension import load
entropy_loss_cuda = load(
    'entropy_loss_cuda', ['./pytorch_entropy_loss/entropy_loss_cuda.cpp', './pytorch_entropy_loss/entropy_loss_cuda_kernel.cu'], verbose=True)
from pytorch_entropy_loss.entropy_loss import EL

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

def entropyLoss(input, minV, maxV):
    return EL.apply(input, minV, maxV)


class EncodeNet(nn.Module):
    def __init__(self):
        super(EncodeNet, self).__init__()

        self.conv_channels_up = nn.Conv2d(1, 64, 1)

        self.conv_down_256_16 = nn.Conv2d(64, 64, 16, 16)
        self.conv_down_128_16 = nn.Conv2d(64, 64, 8, 8)
        self.conv_down_64_16 = nn.Conv2d(64, 64, 4, 4)
        self.conv_down_32_16 = nn.Conv2d(64, 64, 2, 2)

        self.gdn_down_256_16 = pytorch_gdn.GDN(64)
        self.gdn_down_128_16 = pytorch_gdn.GDN(64)
        self.gdn_down_64_16 = pytorch_gdn.GDN(64)
        self.gdn_down_32_16 = pytorch_gdn.GDN(64)

        self.conv_down_256_128 = nn.Conv2d(64, 64, 2, 2)
        self.conv_down_128_64 = nn.Conv2d(64, 64, 2, 2)
        self.conv_down_64_32 = nn.Conv2d(64, 64, 2, 2)
        self.conv_down_32_16 = nn.Conv2d(64, 64, 2, 2)

        self.gdn_down_256_128 = pytorch_gdn.GDN(64)
        self.gdn_down_128_64 = pytorch_gdn.GDN(64)
        self.gdn_down_64_32 = pytorch_gdn.GDN(64)
        self.gdn_down_32_16 = pytorch_gdn.GDN(64)

        self.conv_down_256_32 = nn.Conv2d(64, 64, 8, 8)
        self.conv_down_256_64 = nn.Conv2d(64, 64, 4, 4)
        self.conv_down_128_32 = nn.Conv2d(64, 64, 4, 4)
        self.gdn_down_256_32 = pytorch_gdn.GDN(64)
        self.gdn_down_256_64 = pytorch_gdn.GDN(64)
        self.gdn_down_128_32 = pytorch_gdn.GDN(64)


    def forward(self, x):

        x1 = F.leaky_relu(self.conv_channels_up(x))
        y1 = self.gdn_down_256_16(F.leaky_relu(self.conv_down_256_16(x1)))

        x2 = self.gdn_down_256_128(F.leaky_relu(self.conv_down_256_128(x1)))
        y2 = self.gdn_down_128_16(F.leaky_relu(self.conv_down_128_16(x2)))

        x3 = self.gdn_down_128_64(F.leaky_relu(self.conv_down_128_64(x2)))
        x3 = x3 + self.gdn_down_256_64(F.leaky_relu(self.conv_down_256_64(x1)))
        y3 = self.gdn_down_64_16(F.leaky_relu(self.conv_down_64_16(x3)))

        x4 = self.gdn_down_64_32(F.leaky_relu(self.conv_down_64_32(x3)))
        x4 = x4 + self.gdn_down_256_32(F.leaky_relu(self.conv_down_256_32(x1)))
        x4 = x4 + self.gdn_down_128_32(F.leaky_relu(self.conv_down_128_32(x2)))
        y4 = self.gdn_down_32_16(F.leaky_relu(self.conv_down_32_16(x4)))

        return y1 + y2 + y3 + y4







class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()

        self.tconv_channels_down = nn.ConvTranspose2d(64, 1, 1)

        self.tconv_up_16_256 = nn.ConvTranspose2d(64, 64, 16, 16)
        self.tconv_up_16_128 = nn.ConvTranspose2d(64, 64, 8, 8)
        self.tconv_up_16_64 = nn.ConvTranspose2d(64, 64, 4, 4)
        self.tconv_up_16_32 = nn.ConvTranspose2d(64, 64, 2, 2)

        self.igdn_up_16_256 = pytorch_gdn.GDN(64, True)
        self.igdn_up_16_128 = pytorch_gdn.GDN(64, True)
        self.igdn_up_16_64 = pytorch_gdn.GDN(64, True)
        self.igdn_up_16_32 = pytorch_gdn.GDN(64, True)

        self.tconv_up_32_64 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.tconv_up_64_128 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.tconv_up_128_256 = nn.ConvTranspose2d(64, 64, 2, 2)

        self.igdn_up_32_64 = pytorch_gdn.GDN(64, True)
        self.igdn_up_64_128 = pytorch_gdn.GDN(64, True)
        self.igdn_up_128_256 = pytorch_gdn.GDN(64, True)

        self.tconv_up_32_256 = nn.ConvTranspose2d(64, 64, 8, 8)
        self.tconv_up_32_128 = nn.ConvTranspose2d(64, 64, 4, 4)
        self.tconv_up_64_256 = nn.ConvTranspose2d(64, 64, 4, 4)
        self.igdn_up_32_256 = pytorch_gdn.GDN(64, True)
        self.igdn_up_32_128 = pytorch_gdn.GDN(64, True)
        self.igdn_up_64_256 = pytorch_gdn.GDN(64, True)

    def forward(self, x):
        x4 = F.leaky_relu(self.tconv_up_16_32(self.igdn_up_16_32(x)))

        x3 = F.leaky_relu(self.tconv_up_16_64(self.igdn_up_16_64(x)))
        x3_ = F.leaky_relu(self.tconv_up_32_64(self.igdn_up_32_64(x4)))
        x3 = x3 + x3_

        x2 = F.leaky_relu(self.tconv_up_16_128(self.igdn_up_16_128(x)))
        x2_ = F.leaky_relu(self.tconv_up_64_128(self.igdn_up_64_128(x3)))
        x2 = x2 + x2_
        x2 = x2 + F.leaky_relu(self.tconv_up_32_128(self.igdn_up_32_128(x4)))

        x1 = F.leaky_relu(self.tconv_up_16_256(self.igdn_up_16_256(x)))
        x1_ = F.leaky_relu(self.tconv_up_128_256(self.igdn_up_128_256(x2)))
        x1 = x1 + x1_
        x1 = x1 + F.leaky_relu(self.tconv_up_32_256(self.igdn_up_32_256(x4)))
        x1 = x1 + F.leaky_relu(self.tconv_up_64_256(self.igdn_up_64_256(x3)))

        x = F.leaky_relu(self.tconv_channels_down(x1))

        return x







import bmpReader
'''
argv:
1: 使用哪个显卡
2: 为0则重新开始训练 否则读取之前的模型
3: 学习率 Adam默认是1e-3
4: 训练次数
5: 保存的模型名字
6: λ 训练目标是最小化loss = λ*NLPL + (1-λ)EL
   增大λ 则训练目标向质量方向偏移
7: batchSize
'''

if(len(sys.argv)!=8):
    print('1: 使用哪个显卡\n'
          '2: 为0则重新开始训练 否则读取之前的模型\n'
          '3: 学习率 Adam默认是1e-3\n'
          '4: 训练次数\n'
          '5: 保存的模型标号\n'
          '6: λ 训练目标是最小化loss = λ*MS_SSIM + (1-λ)EL 增大λ 则训练目标向质量方向偏移\n'
          '7: batchSize')
    exit(0)

batchSize = int(sys.argv[7]) # 一次读取?张图片进行训练
dReader = bmpReader.datasetReader(batchSize)
torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡
import nlpDistance




if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    encNet = EncodeNet().cuda()
    decNet = DecodeNet().cuda()
    print('create new model')
else:
    encNet = torch.load('./models/encNet_' + sys.argv[5] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda()
    decNet = torch.load('./models/decNet_' + sys.argv[5] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda()
    print('read ./models/' + sys.argv[5] + '.pkl')

print(encNet)
print(decNet)


MSELoss = nn.MSELoss()

MS_SSIMLambda = float(sys.argv[6])

optimizer = torch.optim.Adam([{'params':encNet.parameters()},{'params':decNet.parameters()}], lr=float(sys.argv[3]))

trainData = torch.empty([batchSize, 1, 256, 256]).float().cuda()



for i in range(int(sys.argv[4])):


    defMaxLossOfTrainData = 0

    for j in range(16): # 每16批 当作一个训练单元 统计这16批数据的表现
        for k in range(batchSize):
            trainData[k] = torch.from_numpy(dReader.readImg()).float().cuda()

        optimizer.zero_grad()
        encData = encNet(trainData)
        qEncData = quantize(encData)
        decData = decNet(qEncData)

        currentMSEL = MSELoss(trainData, decData)
        if(MS_SSIMLambda==0):
            minV = int(qEncData.min().item())
            maxV = int(qEncData.max().item())
            currentEL = entropyLoss(qEncData, minV, maxV)
            currentMS_SSIM = torch.zeros_like(currentEL)


        elif(MS_SSIMLambda==1):
            currentMS_SSIM = pytorch_msssim.ms_ssim(trainData, decData, data_range=255, size_average=True)
            currentEL = torch.zeros_like(currentMS_SSIM)

        else:
            currentMS_SSIM = pytorch_msssim.ms_ssim(trainData, decData, data_range=255, size_average=True)
            minV = int(qEncData.min().item())
            maxV = int(qEncData.max().item())
            currentEL = entropyLoss(qEncData, minV, maxV)





        if(currentMSEL > 500):
            loss = currentMSEL
        else:
            loss = -MS_SSIMLambda * currentMS_SSIM + (1-MS_SSIMLambda) * currentEL

        if(defMaxLossOfTrainData==0):
            maxLossOfTrainData = loss
            maxLossTrainEL = currentEL
            maxLossTrainMSEL = currentMSEL
            maxLossTrainMS_SSIM = currentMS_SSIM
            defMaxLossOfTrainData = 1
        else:
            if(loss>maxLossOfTrainData):
                maxLossOfTrainData = loss # 保存所有训练样本中的最大损失
                maxLossTrainEL = currentEL
                maxLossTrainMSEL = currentMSEL
                maxLossTrainMS_SSIM = currentMS_SSIM

        loss.backward()
        optimizer.step()
        print('%.3f'%loss.item(), ' ', end='')
        sys.stdout.flush()

    if (i == 0):
        minLoss = maxLossOfTrainData
        minLossEL = maxLossTrainEL
        minLossMSEL = maxLossTrainMSEL
        minLossMS_SSIM = maxLossTrainMS_SSIM
    else:
        if (minLoss > maxLossOfTrainData):  # 保存最小loss对应的模型
            minLoss = maxLossOfTrainData
            minLossEL = maxLossTrainEL
            minLossMSEL = maxLossTrainMSEL
            minLossMS_SSIM = maxLossTrainMS_SSIM
            torch.save(encNet, './models/encNet_' + sys.argv[5] + '.pkl')
            torch.save(decNet, './models/decNet_' + sys.argv[5] + '.pkl')
            print('save ./models/' + sys.argv[5] + '.pkl')

    print(sys.argv,end='\n')
    print(i)
    print('本次训练最大loss=','%.3f'%maxLossOfTrainData.item(),'MSEL=','%.3f'%maxLossTrainMSEL.item(),'EL=','%.3f'%maxLossTrainEL.item(),'MS_SSIM=','%.3f'%maxLossTrainMS_SSIM.item())
    print('minLoss=','%.3f'%minLoss.item(),'MSEL=','%.3f'%minLossMSEL.item(),'EL=','%.3f'%minLossEL.item(),'MS_SSIM=','%.3f'%minLossMS_SSIM.item())








