import numpy
import torch
from bitstream import BitStream
import huffmanTable
import pyhuffman.pyhuffman as pyhuffman


def runValueHuffmanEncode(bitStream, inputData):# inputData是一维numpy
    # 对inputData进行霍夫曼编码，先将run编码，再将value编码
    startPos = 0
    length = inputData.shape[0]
    runList = []
    valueList = []
    '''
    形如 0 1 1 2 2 2的数据
    runList = 1 2 3
    valueList = 0 1 2
    '''
    while 1:
        endPos = startPos

        while 1:
            if(endPos >= length):
                endPos = endPos - 1
                break
            if(inputData[endPos]==inputData[startPos]):
                endPos = endPos + 1
            else:
                endPos = endPos - 1
                break # 退出循环时 [startPos,endPos]内元素均相同，且endPos+1位置元素与endPos位置元素不同
        runList.append(endPos - startPos + 1)
        valueList.append(inputData[startPos])
        startPos = endPos + 1
        if(startPos>=length):
            break
    runList = numpy.asarray(runList)
    valueList = numpy.asarray(valueList)
    print('编码runList')
    huffmanEncode(bitStream, runList)
    print('编码valueList')
    huffmanEncode(bitStream, valueList)


def huffmanEncode(bitStream, inputData):# inputData是一维numpy
    minV = inputData.min()
    maxV = inputData.max()
    runListHistc = torch.histc(torch.from_numpy(inputData).cuda(), min=minV, max=maxV,
                               bins=int(maxV - minV + 1)).cpu().numpy()
    print('inputData的最值，分布分别为', minV, maxV, runListHistc)
    pOrder = numpy.argsort(-runListHistc) + minV
    # pOrder保存出现频率从高到低的各个数值
    print('各个数值按出现频率从高到低排列分别为', pOrder)
    vDict = dict.fromkeys(numpy.linspace(minV, maxV, maxV - minV + 1))
    for i in range(pOrder.shape[0]):
        vDict[pOrder[i]] = i
    print('数值-霍夫曼编码表序号的字典为', vDict)
    # vDict[i]保存的是数值i在频率顺序中是第几位，因此数值i对应的霍夫曼编码为encodeTable[vDict[i]]

    # 霍夫曼编码
    # 注意 并不能保证huffmanTable.encodeTable的长度是足够的 但绝大多数情况下 它的长度应该都是足够的
    # 不够时 下面的代码在运行的时候会报错
    for i in range(inputData.shape[0]):
        bitStream.write(huffmanTable.encodeTable[vDict[inputData[i]]], bool)

if __name__ == '__main__':
    exit(0)

