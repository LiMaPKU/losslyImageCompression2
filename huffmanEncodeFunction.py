import numpy
import torch
from bitstream import BitStream
from collections import namedtuple
from collections import OrderedDict
from queue import Queue, PriorityQueue


class binTreeNode:
    def __init__(self, v, f):
        self.left = None
        self.right = None
        self.value = v
        self.frequency = f

    def __lt__(self, other):
        return self.frequency < other.frequency


def traversalTreeCreateTable(treeNode, huffmanTable, binList):
    if(treeNode.left!=None):
        binList.append(0)
        traversalTreeCreateTable(treeNode.left, huffmanTable, binList)
        binList.pop()

    if(treeNode.right!=None):
        binList.append(1)
        traversalTreeCreateTable(treeNode.right, huffmanTable, binList)
        binList.pop()

    if(treeNode.value!=None): # 防止重复访问
        huffmanTable.setdefault(treeNode.value, []).extend(binList)
        treeNode.value = None # 防止重复访问







def createHuffmanTable(inputData):# inputData是一维numpy
    # 获取数据分布
    minV = int(inputData.min())
    maxV = int(inputData.max())
    inputDataHistc = torch.histc(torch.from_numpy(inputData).cuda(), min=minV, max=maxV,
                               bins=(maxV - minV + 1)).cpu().numpy()
    print('inputData的最值，分布分别为', minV, maxV, inputDataHistc)

    vfQueue = PriorityQueue()# 数值-频数 优先队列
    for i in range(minV, maxV + 1):
        if(inputDataHistc[i - minV]!=0):
            vfQueue.put(binTreeNode(i, inputDataHistc[i - minV]))
    # vfQueue中，按照频率从低到高排序，vfQueue.get()是频率最低的
    while True:
        # 选取2个最小的
        leftNode = vfQueue.get()
        rightNode = vfQueue.get()
        newNode = binTreeNode(None, leftNode.frequency + rightNode.frequency)
        newNode.left = leftNode
        newNode.right = rightNode
        vfQueue.put(newNode)
        if(vfQueue.qsize()==1):
            break

    rootNode = vfQueue.get()
    huffmanTable = {} # 字典，key为值，value为二进制码
    binList = []
    traversalTreeCreateTable(rootNode, huffmanTable, binList)
    print('霍夫曼编码表为（类型为python 字典）', huffmanTable)
    return huffmanTable






def runValueHuffmanEncode(bitStream, inputData):# inputData是一维numpy
    # 对inputData进行霍夫曼编码，按run/Value的方式
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
    runValue = namedtuple('runValue',['run','Value'])
    runValueDict = {} # 字典，键为runValue二元组，值为出现的次数
    for i in range(runList.__len__()):
        rv = runValue(runList[i],valueList[i])
        if(rv not in runValueDict.keys()):
            runValueDict[rv] = 1
        else:
            runValueDict[rv] = runValueDict[rv] + 1

    runValueOrderedList = sorted(runValueDict.items(), key=lambda x: x[1], reverse=True)
    # runValueOrderedList[0]到[n]是出现次数递减的run value
    print(runValueOrderedList)
    runValueOrderedDict = OrderedDict()
    for i in range(runValueOrderedList.__len__()):
        runValueOrderedDict[runValueOrderedList[i][0]] = i # 将runValue和次序存进输入，出现次数最多的分配霍夫曼表中的0

    print(runValueOrderedDict)
    # 需要保存编码表，尚未编写--------------------------------

    # -------------------------------------------------------
    # 霍夫曼编码
    # 注意 并不能保证huffmanTable.encodeTable的长度是足够的 但绝大多数情况下 它的长度应该都是足够的
    # 不够时 下面的代码在运行的时候会报错
    for i in range(runList.__len__()):
        rv = runValue(runList[i], valueList[i])
        bitStream.write(huffmanTable.encodeTable[runValueOrderedDict[rv]], bool)


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
    # 需要保存编码表，尚未编写--------------------------------

    # -------------------------------------------------------


    # 霍夫曼编码
    # 注意 并不能保证huffmanTable.encodeTable的长度是足够的 但绝大多数情况下 它的长度应该都是足够的
    # 不够时 下面的代码在运行的时候会报错
    for i in range(inputData.shape[0]):
        bitStream.write(huffmanTable.encodeTable[vDict[inputData[i]]], bool)



if __name__ == '__main__':
    exit(0)

