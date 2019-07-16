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

        self.conv_channels_up = nn.Conv2d(1, 64, 1)

        self.conv_down_256_128 = nn.Conv2d(64, 64, 2, 2)

        self.conv_down_128_64 = nn.Conv2d(64, 64, 2, 2)

        self.conv_down_64_32 = nn.Conv2d(64, 64, 2, 2)

        self.conv_down_32_16 = nn.Conv2d(64, 64, 2, 2)

        self.conv_down_16_8 = nn.Conv2d(64, 64, 2, 2)

        self.gdn_down_256_128 = pytorch_gdn.GDN(64)
        self.gdn_down_128_64 = pytorch_gdn.GDN(64)
        self.gdn_down_64_32 = pytorch_gdn.GDN(64)
        self.gdn_down_32_16 = pytorch_gdn.GDN(64)
        self.gdn_down_16_8 = pytorch_gdn.GDN(64)

    def forward(self, x):

        x = x /255

        x1 = F.leaky_relu(self.conv_channels_up(x))

        x2 = self.gdn_down_256_128(F.leaky_relu(self.conv_down_256_128(x1)))

        x3 = self.gdn_down_128_64(F.leaky_relu(self.conv_down_128_64(x2)))

        x4 = self.gdn_down_64_32(F.leaky_relu(self.conv_down_64_32(x3)))

        x5 = self.gdn_down_32_16(F.leaky_relu(self.conv_down_32_16(x4)))

        x6 = self.gdn_down_16_8(F.leaky_relu(self.conv_down_16_8(x5)))

        return vecNorm(x6)







class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()

        self.tconv_channels_down = nn.ConvTranspose2d(64, 1, 1)

        self.tconv_up_8_16 = nn.ConvTranspose2d(64, 64, 2, 2)

        self.tconv_up_16_32 = nn.ConvTranspose2d(64, 64, 2, 2)

        self.tconv_up_32_64 = nn.ConvTranspose2d(64, 64, 2, 2)

        self.tconv_up_64_128 = nn.ConvTranspose2d(64, 64, 2, 2)

        self.tconv_up_128_256 = nn.ConvTranspose2d(64, 64, 2, 2)

        self.igdn_up_8_16 = pytorch_gdn.GDN(64, True)
        self.igdn_up_16_32 = pytorch_gdn.GDN(64, True)
        self.igdn_up_32_64 = pytorch_gdn.GDN(64, True)
        self.igdn_up_64_128 = pytorch_gdn.GDN(64, True)
        self.igdn_up_128_256 = pytorch_gdn.GDN(64, True)

    def forward(self, x6):

        x5 = F.leaky_relu(self.tconv_up_8_16(self.igdn_up_8_16(x6)))

        x4 = F.leaky_relu(self.tconv_up_16_32(self.igdn_up_16_32(x5)))

        x3 = F.leaky_relu(self.tconv_up_32_64(self.igdn_up_32_64(x4)))

        x2 = F.leaky_relu(self.tconv_up_64_128(self.igdn_up_64_128(x3)))

        x1 = F.leaky_relu(self.tconv_up_128_256(self.igdn_up_128_256(x2)))

        x = F.leaky_relu(self.tconv_channels_down(x1))

        return x*255



class InpaintNet(nn.Module):
    def __init__(self):
        super(InpaintNet, self).__init__()
        self.conv_channels_up = nn.Conv2d(1, 64, 1)
        self.conv_channels_down = nn.Conv2d(64, 1, 1)

        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv9 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv11 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv13 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv14 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv15 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv16 = nn.Conv2d(64, 64, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(64)
        self.bn8 = nn.BatchNorm2d(64)
        self.bn9 = nn.BatchNorm2d(64)
        self.bn10 = nn.BatchNorm2d(64)
        self.bn11 = nn.BatchNorm2d(64)
        self.bn12 = nn.BatchNorm2d(64)
        self.bn13 = nn.BatchNorm2d(64)
        self.bn14 = nn.BatchNorm2d(64)
        self.bn15 = nn.BatchNorm2d(64)
        self.bn16 = nn.BatchNorm2d(64)



    def forward(self, x, n):
        x = x / 255

        x = F.leaky_relu(self.conv_channels_up(x))

        xA = x
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        x = x + xA

        if(n==2):
            x = F.leaky_relu(self.conv_channels_down(x))

            x = x * 255

            return x


        xA = x
        x = self.bn3(F.leaky_relu(self.conv3(x)))
        x = self.bn4(F.leaky_relu(self.conv4(x)))
        x = x + xA

        if(n==4):
            x = F.leaky_relu(self.conv_channels_down(x))

            x = x * 255

            return x

        xA = x
        x = self.bn5(F.leaky_relu(self.conv5(x)))
        x = self.bn6(F.leaky_relu(self.conv6(x)))
        x = x + xA

        if(n==6):
            x = F.leaky_relu(self.conv_channels_down(x))

            x = x * 255

            return x

        xA = x
        x = self.bn7(F.leaky_relu(self.conv7(x)))
        x = self.bn8(F.leaky_relu(self.conv8(x)))
        x = x + xA

        if(n==8):
            x = F.leaky_relu(self.conv_channels_down(x))

            x = x * 255

            return x

        xA = x
        x = self.bn9(F.leaky_relu(self.conv9(x)))
        x = self.bn10(F.leaky_relu(self.conv10(x)))
        x = x + xA

        if(n==10):
            x = F.leaky_relu(self.conv_channels_down(x))

            x = x * 255

            return x

        xA = x
        x = self.bn11(F.leaky_relu(self.conv11(x)))
        x = self.bn12(F.leaky_relu(self.conv12(x)))
        x = x + xA

        if(n==12):
            x = F.leaky_relu(self.conv_channels_down(x))

            x = x * 255

            return x

        xA = x
        x = self.bn13(F.leaky_relu(self.conv13(x)))
        x = self.bn14(F.leaky_relu(self.conv14(x)))
        x = x + xA

        if(n==14):
            x = F.leaky_relu(self.conv_channels_down(x))

            x = x * 255

            return x

        xA = x
        x = self.bn15(F.leaky_relu(self.conv15(x)))
        x = self.bn16(F.leaky_relu(self.conv16(x)))
        x = x + xA

        if(n==16):
            x = F.leaky_relu(self.conv_channels_down(x))

            x = x * 255

            return x



import bmpReader
'''
argv:
1: 使用哪个显卡
2: 为0则重新开始训练 否则读取之前的模型
3: 学习率 Adam默认是1e-3
4: 训练次数
5: 保存的模型名字
6: batchSize
'''

if(len(sys.argv)!=7):
    print('1: 使用哪个显卡\n'
          '2: 为0则重新开始训练 否则读取之前的模型\n'
          '3: 学习率 Adam默认是1e-3\n'
          '4: 训练次数\n'
          '5: 保存的模型标号\n'
          '6: batchSize')
    exit(0)

batchSize = int(sys.argv[6]) # 一次读取?张图片进行训练
dReader = bmpReader.datasetReader(batchSize)
torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡

encNet = torch.load('./models/encNet_14_1_1.pkl', map_location='cuda:' + sys.argv[1]).cuda().eval()
decNet = torch.load('./models/decNet_14_1_1.pkl', map_location='cuda:' + sys.argv[1]).cuda().eval()
print('read ./models/14_1_1.pkl')


if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    iNet = InpaintNet().cuda().train()
    print('create new model')
else:
    iNet = torch.load('./models/iNet_' + sys.argv[5] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda().train()
    print('read ./models/' + sys.argv[5] + '.pkl')
    print(iNet)


MSELoss = nn.MSELoss()


optimizer = torch.optim.Adam([{'params':iNet.parameters()}], lr=float(sys.argv[3]))

trainData = torch.empty([batchSize, 1, 256, 256]).float().cuda()

meanFilter = torch.ones(size=[1,1,7,7]).float().cuda()/49

for i in range(int(sys.argv[4])):


    defMaxLossOfTrainData = 0

    for j in range(16): # 每16批 当作一个训练单元 统计这16批数据的表现
        for k in range(batchSize):
            trainData[k] = torch.from_numpy(dReader.readImg()).float().cuda()


        trainData.requires_grad_(False)
        encData = encNet(trainData)
        qEncData = quantize(encData, 8)
        decData = decNet(qEncData / 7).detach()

        optimizer.zero_grad()
        recData = iNet(decData, 16)



        #currentMSEL = MSELoss(trainData, recData)

        currentMS_SSIM = pytorch_msssim.ms_ssim(trainData, recData, data_range=255, size_average=True)

        currentEdgeMSEL = extendMSE.EdgeMSELoss(trainData, recData)

        loss = -currentMS_SSIM
        if (torch.isnan(loss)):
            loss.zero_()

        if (defMaxLossOfTrainData == 0):
            maxLossOfTrainData = loss.item()
            maxLossTrainMS_SSIM = currentMS_SSIM.item()
            defMaxLossOfTrainData = 1
        else:
            if (loss > maxLossOfTrainData):
                maxLossOfTrainData = loss.item()  # 保存所有训练样本中的最大损失
                maxLossTrainMS_SSIM = currentMS_SSIM.item()

        if (loss.item() > -0.7):
            currentEdgeMSEL.backward()
        else:
            loss.backward()
        optimizer.step()
        print('%.3f' % loss.item(), ' ', end='')
        sys.stdout.flush()

    if (i == 0):
        minLossI = i
        minLoss = maxLossOfTrainData
        minLossMS_SSIM = maxLossTrainMS_SSIM
    else:
        if (minLoss > maxLossOfTrainData):  # 保存最小loss对应的模型
            minLossI = i
            minLoss = maxLossOfTrainData
            minLossMS_SSIM = maxLossTrainMS_SSIM
            # minLossEdgeMSEL = maxLossTrainEdgeMSEL
            torch.save(iNet, './models/iNet_' + sys.argv[5] + '.pkl')
            print('save ./models/' + sys.argv[5] + '.pkl')

    print(sys.argv, end='\n')
    print(i, minLossI)
    print('本次训练最大loss=', '%.3f' % maxLossOfTrainData, 'MS_SSIM=', '%.3f' % maxLossTrainMS_SSIM)
    print('minLoss=', '%.3f' % minLoss, 'MS_SSIM=', '%.3f' % minLossMS_SSIM)







