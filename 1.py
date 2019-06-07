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

        self.conv_channels_up_1_128 = nn.Conv2d(1, 128, 1)

        self.conv128_0 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv128_1 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv128_2 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv128_3 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv128_4 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv128_5 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv128_6 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv128_7 = nn.Conv2d(128, 128, 5, padding=2)

        self.bn128_0 = nn.BatchNorm2d(128)
        self.bn128_1 = nn.BatchNorm2d(128)
        self.bn128_2 = nn.BatchNorm2d(128)
        self.bn128_3 = nn.BatchNorm2d(128)
        self.bn128_4 = nn.BatchNorm2d(128)
        self.bn128_5 = nn.BatchNorm2d(128)
        self.bn128_6 = nn.BatchNorm2d(128)
        self.bn128_7 = nn.BatchNorm2d(128)

        self.bn_A_128_0 = nn.BatchNorm2d(128)
        self.bn_A_128_1 = nn.BatchNorm2d(128)
        self.bn_A_128_2 = nn.BatchNorm2d(128)
        self.bn_A_128_3 = nn.BatchNorm2d(128)

        self.conv_channels_down_128_64 = nn.Conv2d(128, 64, 1)

        self.conv64_0 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv64_1 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv64_2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv64_3 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv64_4 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv64_5 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv64_6 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv64_7 = nn.Conv2d(64, 64, 5, padding=2)

        self.bn64_0 = nn.BatchNorm2d(64)
        self.bn64_1 = nn.BatchNorm2d(64)
        self.bn64_2 = nn.BatchNorm2d(64)
        self.bn64_3 = nn.BatchNorm2d(64)
        self.bn64_4 = nn.BatchNorm2d(64)
        self.bn64_5 = nn.BatchNorm2d(64)
        self.bn64_6 = nn.BatchNorm2d(64)
        self.bn64_7 = nn.BatchNorm2d(64)

        self.bn_A_64_0 = nn.BatchNorm2d(64)
        self.bn_A_64_1 = nn.BatchNorm2d(64)
        self.bn_A_64_2 = nn.BatchNorm2d(64)
        self.bn_A_64_3 = nn.BatchNorm2d(64)

        self.conv_channels_down_64_32 = nn.Conv2d(64, 32, 1)

        self.conv32_0 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv32_1 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv32_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv32_3 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv32_4 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv32_5 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv32_6 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv32_7 = nn.Conv2d(32, 32, 5, padding=2)

        self.bn32_0 = nn.BatchNorm2d(32)
        self.bn32_1 = nn.BatchNorm2d(32)
        self.bn32_2 = nn.BatchNorm2d(32)
        self.bn32_3 = nn.BatchNorm2d(32)
        self.bn32_4 = nn.BatchNorm2d(32)
        self.bn32_5 = nn.BatchNorm2d(32)
        self.bn32_6 = nn.BatchNorm2d(32)
        self.bn32_7 = nn.BatchNorm2d(32)

        self.bn_A_32_0 = nn.BatchNorm2d(32)
        self.bn_A_32_1 = nn.BatchNorm2d(32)
        self.bn_A_32_2 = nn.BatchNorm2d(32)
        self.bn_A_32_3 = nn.BatchNorm2d(32)



    def forward(self, x):

        # -------------------------------------
        # n*1*256*256 -> n*128*256*256
        x = F.leaky_relu(self.conv_channels_up_1_128(x))
        # -------------------------------------

        # -------------------------------------
        # n*128*256*256 多层卷积
        xA = self.bn_A_128_0(x)

        x = F.leaky_relu(self.conv128_0(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn128_0(x)

        x = F.leaky_relu(self.conv128_1(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn128_1(x)

        x = x + xA

        xA = self.bn_A_128_1(x)

        x = F.leaky_relu(self.conv128_2(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn128_2(x)

        x = F.leaky_relu(self.conv128_3(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn128_3(x)

        x = x + xA

        xA = self.bn_A_128_2(x)

        x = F.leaky_relu(self.conv128_4(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn128_4(x)

        x = F.leaky_relu(self.conv128_5(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn128_5(x)

        x = x + xA

        xA = self.bn_A_128_3(x)

        x = F.leaky_relu(self.conv128_6(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn128_6(x)

        x = F.leaky_relu(self.conv128_7(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn128_7(x)

        x = x + xA
        # -------------------------------------

        # -------------------------------------
        # n*128*256*256 -> n*64*256*256
        x = F.leaky_relu(self.conv_channels_down_128_64(x))
        # -------------------------------------

        # -------------------------------------
        # n*64*256*256 多层卷积
        xA = self.bn_A_64_0(x)

        x = F.leaky_relu(self.conv64_0(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn64_0(x)

        x = F.leaky_relu(self.conv64_1(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn64_1(x)

        x = x + xA

        xA = self.bn_A_64_1(x)

        x = F.leaky_relu(self.conv64_2(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn64_2(x)

        x = F.leaky_relu(self.conv64_3(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn64_3(x)

        x = x + xA

        xA = self.bn_A_64_2(x)

        x = F.leaky_relu(self.conv64_4(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn64_4(x)

        x = F.leaky_relu(self.conv64_5(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn64_5(x)

        x = x + xA

        xA = self.bn_A_64_3(x)

        x = F.leaky_relu(self.conv64_6(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn64_6(x)

        x = F.leaky_relu(self.conv64_7(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn64_7(x)

        x = x + xA
        # -------------------------------------

        # -------------------------------------
        # n*64*256*256 -> n*32*256*256
        x = F.leaky_relu(self.conv_channels_down_64_32(x))
        # -------------------------------------

        # -------------------------------------
        # n*32*256*256 多层卷积
        xA = self.bn_A_32_0(x)

        x = F.leaky_relu(self.conv32_0(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn32_0(x)

        x = F.leaky_relu(self.conv32_1(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn32_1(x)

        x = x + xA

        xA = self.bn_A_32_1(x)

        x = F.leaky_relu(self.conv32_2(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn32_2(x)

        x = F.leaky_relu(self.conv32_3(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn32_3(x)

        x = x + xA

        xA = self.bn_A_32_2(x)

        x = F.leaky_relu(self.conv32_4(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn32_4(x)

        x = F.leaky_relu(self.conv32_5(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn32_5(x)

        x = x + xA

        xA = self.bn_A_32_3(x)

        x = F.leaky_relu(self.conv32_6(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn32_6(x)

        x = F.leaky_relu(self.conv32_7(x))
        x = x / (torch.norm(x) + 1e-9)
        x = self.bn32_7(x)

        x = x + xA
        # -------------------------------------


        return x

class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()

        self.tconv32_0 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv32_1 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv32_2 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv32_3 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv32_4 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv32_5 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv32_6 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.tconv32_7 = nn.ConvTranspose2d(32, 32, 5, padding=2)

        self.bn32_0 = nn.BatchNorm2d(32)
        self.bn32_1 = nn.BatchNorm2d(32)
        self.bn32_2 = nn.BatchNorm2d(32)
        self.bn32_3 = nn.BatchNorm2d(32)
        self.bn32_4 = nn.BatchNorm2d(32)
        self.bn32_5 = nn.BatchNorm2d(32)
        self.bn32_6 = nn.BatchNorm2d(32)
        self.bn32_7 = nn.BatchNorm2d(32)

        self.bn_A_32_0 = nn.BatchNorm2d(32)
        self.bn_A_32_1 = nn.BatchNorm2d(32)
        self.bn_A_32_2 = nn.BatchNorm2d(32)
        self.bn_A_32_3 = nn.BatchNorm2d(32)

        self.tconv_channels_up_32_64 = nn.ConvTranspose2d(32, 64, 1)

        self.tconv64_0 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv64_1 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv64_2 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv64_3 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv64_4 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv64_5 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv64_6 = nn.ConvTranspose2d(64, 64, 5, padding=2)
        self.tconv64_7 = nn.ConvTranspose2d(64, 64, 5, padding=2)

        self.bn64_0 = nn.BatchNorm2d(64)
        self.bn64_1 = nn.BatchNorm2d(64)
        self.bn64_2 = nn.BatchNorm2d(64)
        self.bn64_3 = nn.BatchNorm2d(64)
        self.bn64_4 = nn.BatchNorm2d(64)
        self.bn64_5 = nn.BatchNorm2d(64)
        self.bn64_6 = nn.BatchNorm2d(64)
        self.bn64_7 = nn.BatchNorm2d(64)

        self.bn_A_64_0 = nn.BatchNorm2d(64)
        self.bn_A_64_1 = nn.BatchNorm2d(64)
        self.bn_A_64_2 = nn.BatchNorm2d(64)
        self.bn_A_64_3 = nn.BatchNorm2d(64)

        self.tconv_channels_up_64_128 = nn.Conv2d(64, 128, 1)

        self.tconv128_0 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv128_1 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv128_2 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv128_3 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv128_4 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv128_5 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv128_6 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.tconv128_7 = nn.ConvTranspose2d(128, 128, 5, padding=2)

        self.bn128_0 = nn.BatchNorm2d(128)
        self.bn128_1 = nn.BatchNorm2d(128)
        self.bn128_2 = nn.BatchNorm2d(128)
        self.bn128_3 = nn.BatchNorm2d(128)
        self.bn128_4 = nn.BatchNorm2d(128)
        self.bn128_5 = nn.BatchNorm2d(128)
        self.bn128_6 = nn.BatchNorm2d(128)
        self.bn128_7 = nn.BatchNorm2d(128)

        self.bn_A_128_0 = nn.BatchNorm2d(128)
        self.bn_A_128_1 = nn.BatchNorm2d(128)
        self.bn_A_128_2 = nn.BatchNorm2d(128)
        self.bn_A_128_3 = nn.BatchNorm2d(128)

        self.tconv_channels_down_128_1 = nn.ConvTranspose2d(128, 1, 1)



    def forward(self, x):

        # -------------------------------------
        # n*32*256*256 多层卷积
        xA = self.bn_A_32_0(x)

        x = F.leaky_relu(self.tconv32_0(x))
        x = x * torch.norm(x)
        x = self.bn32_0(x)

        x = F.leaky_relu(self.tconv32_1(x))
        x = x * torch.norm(x)
        x = self.bn32_1(x)

        x = x + xA

        xA = self.bn_A_32_1(x)

        x = F.leaky_relu(self.tconv32_2(x))
        x = x * torch.norm(x)
        x = self.bn32_2(x)

        x = F.leaky_relu(self.tconv32_3(x))
        x = x * torch.norm(x)
        x = self.bn32_3(x)

        x = x + xA

        xA = self.bn_A_32_2(x)

        x = F.leaky_relu(self.tconv32_4(x))
        x = x * torch.norm(x)
        x = self.bn32_4(x)

        x = F.leaky_relu(self.tconv32_5(x))
        x = x * torch.norm(x)
        x = self.bn32_5(x)

        x = x + xA

        xA = self.bn_A_32_3(x)

        x = F.leaky_relu(self.tconv32_6(x))
        x = x * torch.norm(x)
        x = self.bn32_6(x)

        x = F.leaky_relu(self.tconv32_7(x))
        x = x * torch.norm(x)
        x = self.bn32_7(x)

        x = x + xA
        # -------------------------------------

        # -------------------------------------
        # n*32*256*256 -> n*64*256*256
        x = F.leaky_relu(self.tconv_channels_up_32_64(x))
        # -------------------------------------

        # -------------------------------------
        # n*64*256*256 多层卷积
        xA = self.bn_A_64_0(x)

        x = F.leaky_relu(self.tconv64_0(x))
        x = x * torch.norm(x)
        x = self.bn64_0(x)

        x = F.leaky_relu(self.tconv64_1(x))
        x = x * torch.norm(x)
        x = self.bn64_1(x)

        x = x + xA

        xA = self.bn_A_64_1(x)

        x = F.leaky_relu(self.tconv64_2(x))
        x = x * torch.norm(x)
        x = self.bn64_2(x)

        x = F.leaky_relu(self.tconv64_3(x))
        x = x * torch.norm(x)
        x = self.bn64_3(x)

        x = x + xA

        xA = self.bn_A_64_2(x)

        x = F.leaky_relu(self.tconv64_4(x))
        x = x * torch.norm(x)
        x = self.bn64_4(x)

        x = F.leaky_relu(self.tconv64_5(x))
        x = x * torch.norm(x)
        x = self.bn64_5(x)

        x = x + xA

        xA = self.bn_A_64_3(x)

        x = F.leaky_relu(self.tconv64_6(x))
        x = x * torch.norm(x)
        x = self.bn64_6(x)

        x = F.leaky_relu(self.tconv64_7(x))
        x = x * torch.norm(x)
        x = self.bn64_7(x)

        x = x + xA
        # -------------------------------------

        # -------------------------------------
        # n*64*256*256 -> n*128*256*256
        x = F.leaky_relu(self.tconv_channels_up_64_128(x))
        # -------------------------------------

        # -------------------------------------
        # n*128*256*256 多层卷积
        xA = self.bn_A_128_0(x)

        x = F.leaky_relu(self.tconv128_0(x))
        x = x * torch.norm(x)
        x = self.bn128_0(x)

        x = F.leaky_relu(self.tconv128_1(x))
        x = x * torch.norm(x)
        x = self.bn128_1(x)

        x = x + xA

        xA = self.bn_A_128_1(x)

        x = F.leaky_relu(self.tconv128_2(x))
        x = x * torch.norm(x)
        x = self.bn128_2(x)

        x = F.leaky_relu(self.tconv128_3(x))
        x = x * torch.norm(x)
        x = self.bn128_3(x)

        x = x + xA

        xA = self.bn_A_128_2(x)

        x = F.leaky_relu(self.tconv128_4(x))
        x = x * torch.norm(x)
        x = self.bn128_4(x)

        x = F.leaky_relu(self.tconv128_5(x))
        x = x * torch.norm(x)
        x = self.bn128_5(x)

        x = x + xA

        xA = self.bn_A_128_3(x)

        x = F.leaky_relu(self.tconv128_6(x))
        x = x * torch.norm(x)
        x = self.bn128_6(x)

        x = F.leaky_relu(self.tconv128_7(x))
        x = x * torch.norm(x)
        x = self.bn128_7(x)

        x = x + xA
        # -------------------------------------

        # -------------------------------------
        # n*128*256*256 -> n*1*256*256
        x = F.leaky_relu(self.tconv_channels_down_128_1(x))
        # -------------------------------------


        return x













'''
argv:
1: 使用哪个显卡
2: 为0则重新开始训练 否则读取之前的模型
3: 学习率 Adam默认是1e-3
4: 训练次数
5: 保存的模型名字
6: λ 训练目标是最小化loss = -λ*SSIM + (1-λ)EL
   增大λ 则训练目标向质量方向偏移
'''

if(len(sys.argv)!=7):
    print('1: 使用哪个显卡\n'
          '2: 为0则重新开始训练 否则读取之前的模型\n'
          '3: 学习率 Adam默认是1e-3\n'
          '4: 训练次数\n'
          '5: 保存的模型标号\n'
          '6: lambda')
    exit(0)
torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡
imgNum = os.listdir('./256bmp').__len__()
imgData = numpy.empty([imgNum,1,256,256])
laplacianData = numpy.empty([imgNum,1,256,256])
for i in range(imgNum):
    img = Image.open('./256bmp/' + str(i) + '.bmp').convert('L')
    imgData[i] = numpy.asarray(img).astype(float).reshape([1,256,256])




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

ssimLambda =  float(sys.argv[6])

optimizer = torch.optim.Adam([{'params':encNet.parameters()},{'params':decNet.parameters()}], lr=float(sys.argv[3]))

batchSize = 2 # 一次读取?张图片进行训练
imgData = torch.from_numpy(imgData).float().cuda()
trainData = torch.empty([batchSize, 1, 256, 256]).float().cuda()



for i in range(int(sys.argv[4])):

    readSeq = torch.randperm(imgNum) # 生成读取的随机序列

    j = 0

    defMaxLossOfTrainData = 0

    while(1):
        if(j==imgNum):
            break
        k = 0
        while(1):
            trainData[k] = imgData[readSeq[j]]
            k = k + 1
            j = j + 1
            if(k==batchSize or j==imgNum):
                break

        optimizer.zero_grad()
        encData = encNet(trainData)
        qEncData = quantize(encData)
        decData = decNet(qEncData)

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

        loss = -ssimLambda *currentSL + (1-ssimLambda) * currentEL
        #print('ssim=', currentSL.item(), 'EL=',currentEL.item(),'loss=',loss.item())
        if(defMaxLossOfTrainData==0):
            maxLossOfTrainData = loss
            maxLossTrainSL = currentSL
            maxLossTrainEL = currentEL
            defMaxLossOfTrainData = 1
        else:
            if(loss>maxLossOfTrainData):
                maxLossOfTrainData = loss # 保存所有训练样本中的最大损失
                maxLossTrainSL = currentSL
                maxLossTrainEL = currentEL

        loss.backward()
        optimizer.step()
        print(j,' ',end='')
        sys.stdout.flush()

    if (i == 0):
        minLoss = maxLossOfTrainData
        minLossSL = maxLossTrainSL
        minLossEL = maxLossTrainEL
    else:
        if (minLoss > maxLossOfTrainData):  # 保存最小loss对应的模型
            minLoss = maxLossOfTrainData
            minLossSL = maxLossTrainSL
            minLossEL = maxLossTrainEL
            torch.save(encNet, './models/encNet_' + sys.argv[5] + '.pkl')
            torch.save(decNet, './models/decNet_' + sys.argv[5] + '.pkl')
            print('save ./models/' + sys.argv[5] + '.pkl')

    print(sys.argv,end='\n')
    print(i)
    print('本次训练最大loss=',maxLossOfTrainData.item(),'SSIM=',maxLossTrainSL.item(),'EL=',maxLossTrainEL.item())
    print('minLoss=',minLoss.item(),'SSIM=',minLossSL.item(),'EL=',minLossEL.item())











