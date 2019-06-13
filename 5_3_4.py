from PIL import Image
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim
import sys
import os
import pytorch_ssim
import pytorch_gdn
# minLoss= 634.3695678710938 MSEL= 634.3695678710938 SSIM= 0.18186593055725098 EL= 0.0
# minLoss= 456.6656188964844 MSEL= 456.6656188964844 SSIM= 0.28569766879081726 EL= 0.0
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

        self.conv_channels_up = nn.Conv2d(1, 32, 1)

        self.conv_down_256_32 = nn.Conv2d(32, 32, 8, 8)
        self.conv_down_128_32 = nn.Conv2d(32, 32, 4, 4)
        self.conv_down_64_32 = nn.Conv2d(32, 32, 2, 2)

        self.gdn_down_256_32 = pytorch_gdn.GDN(32)
        self.gdn_down_128_32 = pytorch_gdn.GDN(32)
        self.gdn_down_64_32 = pytorch_gdn.GDN(32)

        self.conv_down_256_128 = nn.Conv2d(32, 32, 2, 2)
        self.conv_down_128_64 = nn.Conv2d(32, 32, 2, 2)

        self.gdn_down_256_128 = pytorch_gdn.GDN(32)
        self.gdn_down_128_64 = pytorch_gdn.GDN(32)
    def forward(self, x):

        # n*1*256*256 -> n*128*256*256
        x1 = F.leaky_relu(self.conv_channels_up(x))
        y1 = self.gdn_down_256_32(F.leaky_relu(self.conv_down_256_32(x1)))

        x2 = self.gdn_down_256_128(F.leaky_relu(self.conv_down_256_128(x1)))
        y2 = self.gdn_down_128_32(F.leaky_relu(self.conv_down_128_32(x2)))


        x3 = self.gdn_down_128_64(F.leaky_relu(self.conv_down_128_64(x2)))
        y3 = self.gdn_down_64_32(F.leaky_relu(self.conv_down_64_32(x3)))


        return y1 + y2 + y3 # n*128*32*32







class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()

        self.tconv_channels_down = nn.ConvTranspose2d(32, 1, 1)

        self.tconv_up_32_256 = nn.ConvTranspose2d(32, 32, 8, 8)
        self.tconv_up_32_128 = nn.ConvTranspose2d(32, 32, 4, 4)
        self.tconv_up_32_64 = nn.ConvTranspose2d(32, 32, 2, 2)

        self.igdn_up_32_256 = pytorch_gdn.GDN(32, True)
        self.igdn_up_32_128 = pytorch_gdn.GDN(32, True)
        self.igdn_up_32_64 = pytorch_gdn.GDN(32, True)

        self.tconv_up_64_128 = nn.ConvTranspose2d(32, 32, 2, 2)
        self.tconv_up_128_256 = nn.ConvTranspose2d(32, 32, 2, 2)

        self.igdn_up_64_128 = pytorch_gdn.GDN(32, True)
        self.igdn_up_128_256 = pytorch_gdn.GDN(32, True)

    def forward(self, x):

        x3 = F.leaky_relu(self.tconv_up_32_64(self.igdn_up_32_64(x)))

        x2_ = F.leaky_relu(self.tconv_up_64_128(self.igdn_up_64_128(x3)))

        x2 = F.leaky_relu(self.tconv_up_32_128(self.igdn_up_32_128(x)))
        x2 = x2 + x2_

        x1_ = F.leaky_relu(self.tconv_up_128_256(self.igdn_up_128_256(x2)))

        x1 = F.leaky_relu(self.tconv_up_32_256(self.igdn_up_32_256(x)))
        x1 = x1 + x1_

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
6: λ 训练目标是最小化loss = -λ*SSIM + (1-λ)EL
   增大λ 则训练目标向质量方向偏移
7: batchSize
'''

if(len(sys.argv)!=8):
    print('1: 使用哪个显卡\n'
          '2: 为0则重新开始训练 否则读取之前的模型\n'
          '3: 学习率 Adam默认是1e-3\n'
          '4: 训练次数\n'
          '5: 保存的模型标号\n'
          '6: λ 训练目标是最小化loss = -λ*SSIM + (1-λ)EL 增大λ 则训练目标向质量方向偏移\n'
          '7: batchSize')
    exit(0)

batchSize = int(sys.argv[7]) # 一次读取?张图片进行训练
dReader = bmpReader.datasetReader(batchSize)
torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡



if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    encNet = EncodeNet().cuda()
    decNet = DecodeNet().cuda()
    print('create new model')
else:
    encNet = torch.load('./models/encNet_' + sys.argv[5] + '.pkl').cuda()
    decNet = torch.load('./models/decNet_' + sys.argv[5] + '.pkl').cuda()
    print('read ./models/' + sys.argv[5] + '.pkl')

print(encNet)
print(decNet)


SSIMLoss = pytorch_ssim.SSIM()
MSELoss = nn.MSELoss()

ssimLambda = float(sys.argv[6])

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
        if(ssimLambda==0):
            minV = int(qEncData.min().item())
            maxV = int(qEncData.max().item())
            currentEL = entropyLoss(qEncData, minV, maxV)
            currentSL = torch.zeros_like(currentEL)


        elif(ssimLambda==1):
            currentSL = SSIMLoss(decData, trainData)
            currentEL = torch.zeros_like(currentSL)

        else:
            currentSL = SSIMLoss(decData, trainData)
            minV = int(qEncData.min().item())
            maxV = int(qEncData.max().item())
            currentEL = entropyLoss(qEncData, minV, maxV)

        if(currentMSEL > 1000 or currentSL < 0.6):
            loss = currentMSEL
        else:
            loss = -ssimLambda *currentSL + (1-ssimLambda) * currentEL
        #print('ssim=', currentSL.item(), 'EL=',currentEL.item(),'loss=',loss.item())
        if(defMaxLossOfTrainData==0):
            maxLossOfTrainData = loss
            maxLossTrainSL = currentSL
            maxLossTrainEL = currentEL
            maxLossTrainMSEL = currentMSEL
            defMaxLossOfTrainData = 1
        else:
            if(loss>maxLossOfTrainData):
                maxLossOfTrainData = loss # 保存所有训练样本中的最大损失
                maxLossTrainSL = currentSL
                maxLossTrainEL = currentEL
                maxLossTrainMSEL = currentMSEL

        loss.backward()
        optimizer.step()
        print(j, ' ', end='')
        sys.stdout.flush()

    if (i == 0):
        minLoss = maxLossOfTrainData
        minLossSL = maxLossTrainSL
        minLossEL = maxLossTrainEL
        minLossMSEL = maxLossTrainMSEL
    else:
        if (minLoss > maxLossOfTrainData):  # 保存最小loss对应的模型
            minLoss = maxLossOfTrainData
            minLossSL = maxLossTrainSL
            minLossEL = maxLossTrainEL
            minLossMSEL = maxLossTrainMSEL
            torch.save(encNet, './models/encNet_' + sys.argv[5] + '.pkl')
            torch.save(decNet, './models/decNet_' + sys.argv[5] + '.pkl')
            print('save ./models/' + sys.argv[5] + '.pkl')

    print(sys.argv,end='\n')
    print(i)
    print('本次训练最大loss=',maxLossOfTrainData.item(),'MSEL=',maxLossTrainMSEL.item(),'SSIM=',maxLossTrainSL.item(),'EL=',maxLossTrainEL.item())
    print('minLoss=',minLoss.item(),'MSEL=',minLossMSEL.item(),'SSIM=',minLossSL.item(),'EL=',minLossEL.item())








