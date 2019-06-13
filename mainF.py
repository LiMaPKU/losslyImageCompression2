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

def mainFun():
    '''
    argv:
    1: 使用哪个显卡
    2: 为0则重新开始训练 否则读取之前的模型
    3: 学习率 Adam默认是1e-3
    4: 训练次数
    5: 保存的模型名字
    6: λ 训练目标是最小化loss = -λ*SSIM + (1-λ)EL
       增大λ 则训练目标向质量方向偏移
    7: 一次读取的图片数量
    '''

    if (len(sys.argv) != 8):
        print('1: 使用哪个显卡\n'
              '2: 为0则重新开始训练 否则读取之前的模型\n'
              '3: 学习率 Adam默认是1e-3\n'
              '4: 训练次数\n'
              '5: 保存的模型名字\n'
              '6: λ 训练目标是最小化loss = -λ*SSIM + (1-λ)EL 增大λ 则训练目标向质量方向偏移\n'
              '7: 一次读取的图片数量'
              )
        exit(0)
    torch.cuda.set_device(int(sys.argv[1]))  # 设置使用哪个显卡
    imgNum = os.listdir('./256bmp').__len__()
    imgData = numpy.empty([imgNum, 1, 256, 256])

    for i in range(imgNum):
        img = Image.open('./256bmp/' + str(i) + '.bmp').convert('L')
        imgData[i] = numpy.asarray(img).astype(float).reshape([1, 256, 256])

    if (sys.argv[2] == '0'):  # 设置是重新开始 还是继续训练
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

    optimizer = torch.optim.Adam([{'params': encNet.parameters()}, {'params': decNet.parameters()}],
                                 lr=float(sys.argv[3]))

    batchSize = int(sys.argv[8])  # 一次读取?张图片进行训练
    imgData = torch.from_numpy(imgData).float().cuda()
    trainData = torch.empty([batchSize, 1, 256, 256]).float().cuda()

    for i in range(int(sys.argv[4])):

        readSeq = torch.randperm(imgNum)  # 生成读取的随机序列

        j = 0

        defMaxLossOfTrainData = 0

        while (1):
            if (j == imgNum):
                break
            k = 0
            while (1):
                trainData[k] = imgData[readSeq[j]]
                k = k + 1
                j = j + 1
                if (k == batchSize or j == imgNum):
                    break

            optimizer.zero_grad()
            encData = encNet(trainData)
            qEncData = quantize(encData)
            decData = decNet(qEncData)

            currentMSEL = MSELoss(trainData, decData)
            if (ssimLambda == 0):
                minV = int(qEncData.min().item())
                maxV = int(qEncData.max().item())
                currentEL = entropyLoss(qEncData, minV, maxV)
                currentSL = torch.zeros_like(currentEL)


            elif (ssimLambda == 1):
                currentSL = SSIMLoss(decData, trainData)
                currentEL = torch.zeros_like(currentSL)

            else:
                currentSL = SSIMLoss(decData, trainData)
                minV = int(qEncData.min().item())
                maxV = int(qEncData.max().item())
                currentEL = entropyLoss(qEncData, minV, maxV)

            if (currentMSEL > 1000):
                loss = currentMSEL
            else:
                loss = -ssimLambda * currentSL + (1 - ssimLambda) * currentEL
            # print('ssim=', currentSL.item(), 'EL=',currentEL.item(),'loss=',loss.item())
            if (defMaxLossOfTrainData == 0):
                maxLossOfTrainData = loss
                maxLossTrainSL = currentSL
                maxLossTrainEL = currentEL
                maxLossTrainMSEL = currentMSEL
                defMaxLossOfTrainData = 1
            else:
                if (loss > maxLossOfTrainData):
                    maxLossOfTrainData = loss  # 保存所有训练样本中的最大损失
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

        print(sys.argv, end='\n')
        print(i)
        print('本次训练最大loss=', maxLossOfTrainData.item(), 'MSEL=', maxLossTrainMSEL.item(), 'SSIM=',
              maxLossTrainSL.item(), 'EL=', maxLossTrainEL.item())
        print('minLoss=', minLoss.item(), 'MSEL=', minLossMSEL.item(), 'SSIM=', minLossSL.item(), 'EL=',
              minLossEL.item())