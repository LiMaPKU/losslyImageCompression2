import numpy
import torch
from bitstream import BitStream
import huffmanTable
#hexToBytes('F0') = 1111 1111 0000 0000(bytes)
def hexToBytes(hexStr):
    num = len(hexStr)//2
    ret = numpy.zeros([num],dtype=int)
    for i in range(num):
        ret[i] = int(hexStr[2*i:2*i+2],16)

    ret = ret.tolist()
    ret = bytes(ret)
    return ret

def encodeDModeList(bitStream, dModeList):
    # 对保存众数前向一阶差分的列表，进行霍夫曼编码

    # 统计数据分布情况
    minV = dModeList.min()
    maxV = dModeList.max()
    dModeListHistc = torch.histc(torch.from_numpy(dModeList).cuda(), min=minV, max=maxV,bins=int(maxV - minV + 1)).cpu().numpy()
    print('dModeList的最值，分布分别为',minV,maxV,dModeListHistc)
    pOrder = numpy.argsort(-dModeListHistc) + minV
    # pOrder保存出现频率从高到低的各个数值
    print('各个数值按出现频率从高到低排列分别为',pOrder)
    vDict = dict.fromkeys(numpy.linspace(minV, maxV, maxV - minV + 1))
    for i in range(pOrder.shape[0]):
        vDict[pOrder[i]] = i
    print('数值-霍夫曼编码表序号的字典为',vDict)
    # vDict[i]保存的是数值i在频率顺序中是第几位，因此数值i对应的霍夫曼编码为encodeTable[vDict[i]]

    # 霍夫曼编码
    for i in range(dModeList.shape[0]):
        bitStream.write(huffmanTable.encodeTable[ vDict[dModeList[i]] ], bool)

    # 写入FF，当作分隔符
    bitStream.write([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], bool)


def encodeDModeList(bitStream, inputData):
    inputDataHistc = torch.histc(torch.from_numpy(inputData).cuda(), min=-3, max=4,bins=8).cpu().numpy()
    print('数据分布为',inputDataHistc)
    pOrder = numpy.argsort(-inputDataHistc) - 3
    # pOrder保存出现频率从高到低的各个数值
    print('各个数值按出现频率从高到低排列分别为', pOrder)
    vDict = dict.fromkeys(numpy.linspace(-3, 4, 8))
    for i in range(pOrder.shape[0]):
        vDict[pOrder[i]] = i
    print('数值-霍夫曼编码表序号的字典为', vDict)
    # vDict[i]保存的是数值i在频率顺序中是第几位，因此数值i对应的霍夫曼编码为encodeTable[vDict[i]]
    for i in range(inputData.shape[0]):
        zCode = inputData[i].reshape([256])[huffmanTable.zigzagOrder]  # z形扫描
        # zCode[0]不参与编码，因为众数已经被保存过到modeList了
