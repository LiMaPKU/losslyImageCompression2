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
        self.conv_down_16_8 = nn.Conv2d(64, 64, 2, 2)

        self.gdn_down_256_128 = pytorch_gdn.GDN(64)
        self.gdn_down_128_64 = pytorch_gdn.GDN(64)
        self.gdn_down_64_32 = pytorch_gdn.GDN(64)
        self.gdn_down_32_16 = pytorch_gdn.GDN(64)
        self.gdn_down_16_8 = pytorch_gdn.GDN(64)

    def forward(self, x, n=5):

        x = F.leaky_relu(self.conv_channels_up(x))
        if(n>=2):
            x = self.gdn_down_256_128(F.leaky_relu(self.conv_down_256_128(x)))
            if(n>=3):
                x = self.gdn_down_128_64(F.leaky_relu(self.conv_down_128_64(x)))
                if(n>=4):
                    x = self.gdn_down_64_32(F.leaky_relu(self.conv_down_64_32(x)))
                    if(n>=5):
                        x = self.gdn_down_32_16(F.leaky_relu(self.conv_down_32_16(x)))
                        if (n >=6 ):
                            x = self.gdn_down_16_8(F.leaky_relu(self.conv_down_16_8(x)))
        return x







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

    def forward(self, x, n=5):

        if(n>=6):
            x = F.leaky_relu(self.tconv_up_8_16(self.igdn_up_8_16(x)))
        if(n>=5):
            x = F.leaky_relu(self.tconv_up_16_32(self.igdn_up_16_32(x)))
        if(n>=4):
            x = F.leaky_relu(self.tconv_up_32_64(self.igdn_up_32_64(x)))
        if(n>=3):
            x = F.leaky_relu(self.tconv_up_64_128(self.igdn_up_64_128(x)))
        if(n>=2):
            x = F.leaky_relu(self.tconv_up_128_256(self.igdn_up_128_256(x)))

        x = F.leaky_relu(self.tconv_channels_down(x))

        return x

def freezeLayer(torchModel):
    for param in torchModel.parameters():
        param.require_grad = False

def unfreezeLayer(torchModel):
    for param in torchModel.parameters():
        param.require_grad = True




import bmpReader
if(len(sys.argv)==5):
    # 测试 参数分别为 哪个显卡 模型名字 图片数量 n
    torch.cuda.set_device(int(sys.argv[1]))  # 设置使用哪个显卡
    encNet = torch.load('./models/encNet_' + sys.argv[2] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda()
    decNet = torch.load('./models/decNet_' + sys.argv[2] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda()
    print('read ./models/' + sys.argv[2] + '.pkl')
    n = int(sys.argv[4])
    dReader = bmpReader.datasetReader(1)
    testData = torch.empty([1, 1, 256, 256]).float().cuda()
    MSELoss = nn.MSELoss()
    sumMSEL = torch.zeros([1]).float().cuda()
    sumMS_SSIM = torch.zeros([1]).float().cuda()
    testNum = int(sys.argv[3])
    for k in range(testNum):
        testData[0] = torch.from_numpy(dReader.readImg()).float().cuda()
        encData = encNet(testData, n)
        if (n == 6):
            qEncData = quantize(encData)
            decData = decNet(qEncData)
        else:
            decData = decNet(encData, n)
        currentMSEL = MSELoss(testData, decData)
        currentMS_SSIM = pytorch_msssim.ms_ssim(testData, decData, data_range=255, size_average=True)
        print(currentMSEL.item(), currentMS_SSIM.item())
        sumMSEL = sumMSEL + currentMSEL
        sumMS_SSIM = sumMS_SSIM + currentMS_SSIM

    print(testNum,'张图片的平均性能为')
    print(sumMSEL.item()/testNum, sumMS_SSIM.item()/testNum)


    exit(0)

'''
argv:
1: 使用哪个显卡
2: 为0则重新开始训练 否则读取之前的模型
3: 学习率 Adam默认是1e-3
4: 训练次数
5: 保存的模型名字
6: batchSize
'''

if(len(sys.argv)!=8):
    print('1: 使用哪个显卡\n'
          '2: 为0则重新开始训练 否则读取之前的模型\n'
          '3: 学习率 Adam默认是1e-3\n'
          '4: 训练次数\n'
          '5: 保存的模型标号\n'
          '6: batchSize\n'
          '7: 本次训练哪一层')
    exit(0)

batchSize = int(sys.argv[6]) # 一次读取?张图片进行训练
dReader = bmpReader.datasetReader(batchSize)
torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡




if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    encNet = EncodeNet().cuda()
    decNet = DecodeNet().cuda()
    print('create new model')
else:
    encNet = torch.load('./models/encNet_' + sys.argv[5] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda()
    decNet = torch.load('./models/decNet_' + sys.argv[5] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda()
    print('read ./models/' + sys.argv[5] + '.pkl')



MSELoss = nn.MSELoss()



trainData = torch.empty([batchSize, 1, 256, 256]).float().cuda()
n = int(sys.argv[7])  # n=1,2,3,4,5

freezeLayer(encNet)
freezeLayer(decNet)

if( n==1 ):
    optimizer = torch.optim.Adam([{'params':encNet.conv_channels_up.parameters()},{'params':decNet.tconv_channels_down.parameters()}], lr=float(sys.argv[3]))
    unfreezeLayer(encNet.conv_channels_up)
    unfreezeLayer(decNet.tconv_channels_down)
if( n==2 ):
    optimizer = torch.optim.Adam([{'params': encNet.conv_down_256_128.parameters()}, {'params': encNet.gdn_down_256_128.parameters()},
                                  {'params': decNet.tconv_up_128_256.parameters()},  {'params': decNet.igdn_up_128_256.parameters()}],lr=float(sys.argv[3]))
    unfreezeLayer(encNet.conv_down_256_128)
    unfreezeLayer(encNet.gdn_down_256_128)
    unfreezeLayer(decNet.tconv_up_128_256)
    unfreezeLayer(decNet.igdn_up_128_256)
if ( n==3 ):
    optimizer =  torch.optim.Adam([{'params': encNet.conv_down_128_64.parameters()}, {'params': encNet.gdn_down_128_64.parameters()},
                                   {'params': decNet.tconv_up_64_128.parameters()},  {'params': decNet.igdn_up_64_128.parameters()}],lr=float(sys.argv[3]))
    unfreezeLayer(encNet.conv_down_128_64)
    unfreezeLayer(encNet.gdn_down_128_64)
    unfreezeLayer(decNet.tconv_up_64_128)
    unfreezeLayer(decNet.igdn_up_64_128)
if (n == 4):
    optimizer = torch.optim.Adam([{'params': encNet.conv_down_64_32.parameters()}, {'params': encNet.gdn_down_64_32.parameters()},
                                  {'params': decNet.tconv_up_32_64.parameters()},  {'params': decNet.igdn_up_32_64.parameters()}],lr=float(sys.argv[3]))
    unfreezeLayer(encNet.conv_down_64_32)
    unfreezeLayer(encNet.gdn_down_64_32)
    unfreezeLayer(decNet.tconv_up_32_64)
    unfreezeLayer(decNet.igdn_up_32_64)
if (n == 5):
    optimizer = torch.optim.Adam([{'params': encNet.conv_down_32_16.parameters()}, {'params': encNet.gdn_down_32_16.parameters()},
                                  {'params': decNet.tconv_up_16_32.parameters()},  {'params': decNet.igdn_up_16_32.parameters()}],lr=float(sys.argv[3]))
    unfreezeLayer(encNet.conv_down_32_16)
    unfreezeLayer(encNet.gdn_down_32_16)
    unfreezeLayer(decNet.tconv_up_16_32)
    unfreezeLayer(decNet.igdn_up_16_32)
if (n == 6):
    optimizer = torch.optim.Adam([{'params': encNet.conv_down_16_8.parameters()}, {'params': encNet.gdn_down_16_8.parameters()},
                                  {'params': decNet.tconv_up_8_16.parameters()},  {'params': decNet.igdn_up_8_16.parameters()}],lr=float(sys.argv[3]))
    unfreezeLayer(encNet.conv_down_16_8)
    unfreezeLayer(encNet.gdn_down_16_8)
    unfreezeLayer(decNet.tconv_up_8_16)
    unfreezeLayer(decNet.igdn_up_8_16)


print(encNet)
print(decNet)

for i in range(int(sys.argv[4])):


    defMaxLossOfTrainData = 0

    for j in range(16): # 每16批 当作一个训练单元 统计这16批数据的表现

        for k in range(batchSize):
            trainData[k] = torch.from_numpy(dReader.readImg()).float().cuda()

        optimizer.zero_grad()

        encData = encNet(trainData, n)
        if(n==6):
            qEncData = quantize(encData)
            decData = decNet(qEncData)
        else:
            decData = decNet(encData, n)

        currentMSEL = MSELoss(trainData, decData)

        currentMS_SSIM = pytorch_msssim.ms_ssim(trainData, decData, data_range=255, size_average=True)

        #currentEdgeMSEL = extendMSE.EdgeMSELoss(trainData, decData)


        loss = currentMSEL

        loss.backward()
        optimizer.step()
        print('%.3f'%currentMS_SSIM.item(),'%.3f'%currentMSEL.item(), ' ', end='')
        sys.stdout.flush()


        if(defMaxLossOfTrainData==0):
            maxLossOfTrainData = loss
            #maxLossTrainMSEL = currentMSEL
            #maxLossTrainMS_SSIM = currentMS_SSIM
            #maxLossTrainEdgeMSEL = currentEdgeMSEL
            defMaxLossOfTrainData = 1
        else:
            if(loss>maxLossOfTrainData):
                maxLossOfTrainData = loss # 保存所有训练样本中的最大损失
                #maxLossTrainMSEL = currentMSEL
                #maxLossTrainMS_SSIM = currentMS_SSIM
                #maxLossTrainEdgeMSEL = currentEdgeMSEL


    if (i == 0):
        minLoss = maxLossOfTrainData
        #minLossMSEL = maxLossTrainMSEL
        #minLossMS_SSIM = maxLossTrainMS_SSIM
        #minLossEdgeMSEL = maxLossTrainEdgeMSEL
    else:
        if (minLoss > maxLossOfTrainData):  # 保存最小loss对应的模型
            minLoss = maxLossOfTrainData
            #minLossMSEL = maxLossTrainMSEL
            #minLossMS_SSIM = maxLossTrainMS_SSIM
            #minLossEdgeMSEL = maxLossTrainEdgeMSEL
            torch.save(encNet, './models/encNet_' + sys.argv[5] + '.pkl')
            torch.save(decNet, './models/decNet_' + sys.argv[5] + '.pkl')
            print('save ./models/' + sys.argv[5] + '.pkl')

    print(sys.argv, end='\n')
    print(i, n)
    print('本次训练最大loss=','%.3f'%maxLossOfTrainData.item())
    print('minLoss=','%.3f'%minLoss.item())







