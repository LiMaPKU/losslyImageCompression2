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
# minLoss= -0.5456319451332092 MSEL= 159.0536651611328 SSIM= 0.5456319451332092 EL= 0.0
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

        self.conv_channels_up = nn.Conv2d(1, 128, 1)

        self.conv1_0 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1_3 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv2_0 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_3 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv_g1 = nn.Conv2d(128, 128, 8, 8)
        self.conv_g2 = nn.Conv2d(128, 128, 4, 4)
        self.conv_g3 = nn.Conv2d(128, 128, 2, 2)

        self.gdn_g1 = pytorch_gdn.GDN(128)
        self.gdn_g2 = pytorch_gdn.GDN(128)
        self.gdn_g3 = pytorch_gdn.GDN(128)

        self.conv_f1 = nn.Conv2d(128, 128, 2, 2)
        self.conv_f2 = nn.Conv2d(128, 128, 2, 2)

        self.gdn_f1 = pytorch_gdn.GDN(128)
        self.gdn_f2 = pytorch_gdn.GDN(128)
    def forward(self, x):

        # n*1*256*256 -> n*128*256*256
        x1 = F.leaky_relu(self.conv_channels_up(x))
        y1 = self.gdn_g1(F.leaky_relu(self.conv_g1(x1)))

        x1A = x1
        x1 = F.leaky_relu(self.conv1_0(x1))
        x1 = F.leaky_relu(self.conv1_1(x1))
        x1 = F.leaky_relu(self.conv1_2(x1))
        x1 = F.leaky_relu(self.conv1_3(x1))
        x1 = x1 + x1A

        x2 = self.gdn_f1(F.leaky_relu(self.conv_f1(x1)))
        y2 = self.gdn_g2(F.leaky_relu(self.conv_g2(x2)))

        x2A = x2
        x2 = F.leaky_relu(self.conv2_0(x2))
        x2 = F.leaky_relu(self.conv2_1(x2))
        x2 = F.leaky_relu(self.conv2_2(x2))
        x2 = F.leaky_relu(self.conv2_3(x2))
        x2 = x2 + x2A

        x3 = self.gdn_f2(F.leaky_relu(self.conv_f2(x2)))
        y3 = self.gdn_g3(F.leaky_relu(self.conv_g3(x3)))




        return y1 + y2 + y3 # n*128*32*32

class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()

        self.tconv_channels_down = nn.ConvTranspose2d(128, 1, 1)

        self.tconv1_0 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.tconv1_1 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.tconv1_2 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.tconv1_3 = nn.ConvTranspose2d(128, 128, 3, padding=1)

        self.tconv2_0 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.tconv2_1 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.tconv2_2 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.tconv2_3 = nn.ConvTranspose2d(128, 128, 3, padding=1)

        self.tconv_g1 = nn.ConvTranspose2d(128, 128, 8, 8)
        self.tconv_g2 = nn.ConvTranspose2d(128, 128, 4, 4)
        self.tconv_g3 = nn.ConvTranspose2d(128, 128, 2, 2)

        self.igdn_g1 = pytorch_gdn.GDN(128, True)
        self.igdn_g2 = pytorch_gdn.GDN(128, True)
        self.igdn_g3 = pytorch_gdn.GDN(128, True)

        self.tconv_f1 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.tconv_f2 = nn.ConvTranspose2d(128, 128, 2, 2)

        self.igdn_f1 = pytorch_gdn.GDN(128, True)
        self.igdn_f2 = pytorch_gdn.GDN(128, True)

    def forward(self, x):

        x3 = F.leaky_relu(self.tconv_g3(self.igdn_g3(x)))
        x3 = F.leaky_relu(self.tconv_f2(self.igdn_f2(x3)))
        x3A = x3
        x3 = F.leaky_relu(self.tconv2_0(x3))
        x3 = F.leaky_relu(self.tconv2_1(x3))
        x3 = F.leaky_relu(self.tconv2_2(x3))
        x3 = F.leaky_relu(self.tconv2_3(x3))
        x3 = x3 + x3A

        x2 = x3 + F.leaky_relu(self.tconv_g2(self.igdn_g2(x)))
        x2 = F.leaky_relu(self.tconv_f1(self.igdn_f1(x2)))
        x2A = x2
        x2 = F.leaky_relu(self.tconv1_0(x2))
        x2 = F.leaky_relu(self.tconv1_1(x2))
        x2 = F.leaky_relu(self.tconv1_2(x2))
        x2 = F.leaky_relu(self.tconv1_3(x2))
        x2 = x2 + x2A

        x1 = x2 + F.leaky_relu(self.tconv_g1(self.igdn_g1(x)))

        x = F.leaky_relu(self.tconv_channels_down(x1))


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
MSELoss = nn.MSELoss()

ssimLambda = float(sys.argv[6])

optimizer = torch.optim.Adam([{'params':encNet.parameters()},{'params':decNet.parameters()}], lr=float(sys.argv[3]))

batchSize = 8 # 一次读取?张图片进行训练
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

        if(currentSL.item() < 0.5):
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
        print(j,' ',end='')
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








