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

        self.conv_channels_up = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv8 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv9 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv11 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv12 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv13 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv14 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv15 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv16 = nn.Conv2d(32, 32, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)
        self.bn8 = nn.BatchNorm2d(32)
        self.bn9 = nn.BatchNorm2d(32)
        self.bn10 = nn.BatchNorm2d(32)
        self.bn11 = nn.BatchNorm2d(32)
        self.bn12 = nn.BatchNorm2d(32)
        self.bn13 = nn.BatchNorm2d(32)
        self.bn14 = nn.BatchNorm2d(32)
        self.bn15 = nn.BatchNorm2d(32)
        self.bn16 = nn.BatchNorm2d(32)




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


        x = F.leaky_relu(self.conv_channels_up(x))

        xA = x
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        x = x + xA

        xA = x
        x = self.bn3(F.leaky_relu(self.conv3(x)))
        x = self.bn4(F.leaky_relu(self.conv4(x)))
        x = x + xA

        xA = x
        x = self.bn5(F.leaky_relu(self.conv5(x)))
        x = self.bn6(F.leaky_relu(self.conv6(x)))
        x = x + xA

        xA = x
        x = self.bn7(F.leaky_relu(self.conv7(x)))
        x = self.bn8(F.leaky_relu(self.conv8(x)))
        x = x + xA

        xA = x
        x = self.bn9(F.leaky_relu(self.conv9(x)))
        x = self.bn10(F.leaky_relu(self.conv10(x)))
        x = x + xA

        xA = x
        x = self.bn11(F.leaky_relu(self.conv11(x)))
        x = self.bn12(F.leaky_relu(self.conv12(x)))
        x = x + xA

        xA = x
        x = self.bn13(F.leaky_relu(self.conv13(x)))
        x = self.bn14(F.leaky_relu(self.conv14(x)))
        x = x + xA

        xA = x
        x = self.bn15(F.leaky_relu(self.conv15(x)))
        x = self.bn16(F.leaky_relu(self.conv16(x)))
        x1 = x + xA



        x2 = self.gdn_down_256_128(F.leaky_relu(self.conv_down_256_128(x1)))

        x3 = self.gdn_down_128_64(F.leaky_relu(self.conv_down_128_64(x2)))

        x4 = self.gdn_down_64_32(F.leaky_relu(self.conv_down_64_32(x3)))

        x5 = self.gdn_down_32_16(F.leaky_relu(self.conv_down_32_16(x4)))

        return vecNorm(x5)







class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()

        self.tconv_channels_down = nn.ConvTranspose2d(32, 1, 5, padding=2)
        self.tconv1 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv2 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv3 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv4 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv5 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv6 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv7 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv8 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv9 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv10 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv11 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv12 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv13 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv14 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv15 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.tconv16 = nn.ConvTranspose2d(32, 32, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)
        self.bn8 = nn.BatchNorm2d(32)
        self.bn9 = nn.BatchNorm2d(32)
        self.bn10 = nn.BatchNorm2d(32)
        self.bn11 = nn.BatchNorm2d(32)
        self.bn12 = nn.BatchNorm2d(32)
        self.bn13 = nn.BatchNorm2d(32)
        self.bn14 = nn.BatchNorm2d(32)
        self.bn15 = nn.BatchNorm2d(32)
        self.bn16 = nn.BatchNorm2d(32)

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

        x = F.leaky_relu(self.tconv_up_128_256(self.igdn_up_128_256(x2)))


        xA = x
        x = self.bn1(F.leaky_relu(self.tconv1(x)))
        x = self.bn2(F.leaky_relu(self.tconv2(x)))
        x = x + xA

        xA = x
        x = self.bn3(F.leaky_relu(self.tconv3(x)))
        x = self.bn4(F.leaky_relu(self.tconv4(x)))
        x = x + xA

        xA = x
        x = self.bn5(F.leaky_relu(self.tconv5(x)))
        x = self.bn6(F.leaky_relu(self.tconv6(x)))
        x = x + xA

        xA = x
        x = self.bn7(F.leaky_relu(self.tconv7(x)))
        x = self.bn8(F.leaky_relu(self.tconv8(x)))
        x = x + xA

        xA = x
        x = self.bn9(F.leaky_relu(self.tconv9(x)))
        x = self.bn10(F.leaky_relu(self.tconv10(x)))
        x = x + xA

        xA = x
        x = self.bn11(F.leaky_relu(self.tconv11(x)))
        x = self.bn12(F.leaky_relu(self.tconv12(x)))
        x = x + xA

        xA = x
        x = self.bn13(F.leaky_relu(self.tconv13(x)))
        x = self.bn14(F.leaky_relu(self.tconv14(x)))
        x = x + xA

        xA = x
        x = self.bn15(F.leaky_relu(self.tconv15(x)))
        x = self.bn16(F.leaky_relu(self.tconv16(x)))
        x = x + xA

        x = F.leaky_relu(self.tconv_channels_down(x))

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




if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    encNet = EncodeNet().cuda().train()
    decNet = DecodeNet().cuda().train()
    print('create new model')
else:
    encNet = torch.load('./models/encNet_' + sys.argv[5] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda().train()
    decNet = torch.load('./models/decNet_' + sys.argv[5] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda().train()
    print('read ./models/' + sys.argv[5] + '.pkl')





optimizer = torch.optim.Adam([{'params':encNet.parameters()},{'params':decNet.parameters()}], lr=float(sys.argv[3]))

trainData = torch.empty([batchSize, 1, 256, 256]).float().cuda()



for i in range(int(sys.argv[4])):


    defMaxLossOfTrainData = 0

    for j in range(16): # 每16批 当作一个训练单元 统计这16批数据的表现
        for k in range(batchSize):
            trainData[k] = torch.from_numpy(dReader.readImg()).float().cuda()

        optimizer.zero_grad()
        encData = encNet(trainData)
        qEncData = quantize(encData, 16)
        decData = decNet(qEncData / 15)

        #currentMSEL = MSELoss(trainData, decData)

        currentMS_SSIM = pytorch_msssim.ms_ssim(trainData, decData, data_range=255, size_average=True)
        currentEdgeMSEL = extendMSE.EdgeMSELoss(trainData, decData)

        loss = -currentMS_SSIM
        if(torch.isnan(loss)):
            loss.zero_()




        if(defMaxLossOfTrainData==0):
            maxLossOfTrainData = loss.item()
            maxLossTrainMS_SSIM = currentMS_SSIM.item()
            defMaxLossOfTrainData = 1
        else:
            if(loss>maxLossOfTrainData):
                maxLossOfTrainData = loss.item() # 保存所有训练样本中的最大损失
                maxLossTrainMS_SSIM = currentMS_SSIM.item()

        if(loss.item()>-0.7):
            currentEdgeMSEL.backward()
        else:
            loss.backward()
        optimizer.step()
        print('%.3f'%loss.item(), ' ', end='')
        sys.stdout.flush()

    if (i == 0):
        minLoss = maxLossOfTrainData
        minLossMS_SSIM = maxLossTrainMS_SSIM
    else:
        if (minLoss > maxLossOfTrainData):  # 保存最小loss对应的模型
            minLoss = maxLossOfTrainData
            minLossMS_SSIM = maxLossTrainMS_SSIM
            #minLossEdgeMSEL = maxLossTrainEdgeMSEL
            torch.save(encNet, './models/encNet_' + sys.argv[5] + '.pkl')
            torch.save(decNet, './models/decNet_' + sys.argv[5] + '.pkl')
            print('save ./models/' + sys.argv[5] + '.pkl')

    print(sys.argv,end='\n')
    print(i)
    print('本次训练最大loss=','%.3f'%maxLossOfTrainData,'MS_SSIM=','%.3f'%maxLossTrainMS_SSIM)
    print('minLoss=','%.3f'%minLoss,'MS_SSIM=','%.3f'%minLossMS_SSIM)







