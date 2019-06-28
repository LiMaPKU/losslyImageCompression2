import numpy
import torch
from bitstream import BitStream
from collections import namedtuple
from collections import OrderedDict
from queue import Queue, PriorityQueue

runValue = namedtuple('runValue', ['run', 'Value'])


class binTreeNode:
    def __init__(self, rv, f):
        self.left = None
        self.right = None
        self.runValue = rv
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

    if(treeNode.runValue!=None): # 防止重复访问
        huffmanTable.setdefault(treeNode.runValue, []).extend(binList)
        treeNode.value = None # 防止重复访问







def runValueHuffmanEncode(inputData, bitStream):# inputData是一维numpy

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
            if (endPos >= length):
                endPos = endPos - 1
                break
            if (inputData[endPos] == inputData[startPos]):
                endPos = endPos + 1
            else:
                endPos = endPos - 1
                break  # 退出循环时 [startPos,endPos]内元素均相同，且endPos+1位置元素与endPos位置元素不同
        runList.append(endPos - startPos + 1)
        valueList.append(inputData[startPos])
        startPos = endPos + 1
        if (startPos >= length):
            break
    runValueDict = {}  # 字典，键为runValue二元组，值为出现的次数
    for i in range(runList.__len__()):
        rv = runValue(runList[i], valueList[i])
        if (rv not in runValueDict.keys()):
            runValueDict[rv] = 1
        else:
            runValueDict[rv] = runValueDict[rv] + 1


    vfQueue = PriorityQueue()# 数值-频数 优先队列
    for key in runValueDict.keys():
        vfQueue.put(binTreeNode(key, runValueDict[key]))

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
    # todo: 保存霍夫曼表到文件---------------------------------------------------------
    for i in range(runList.__len__()):
        bitStream.write(huffmanTable[runValue(runList[i], valueList[i])], bool)

def valueHuffmanEncode(inputData, bitStream):# inputData是一维numpy
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
    # todo: 保存霍夫曼表到文件---------------------------------------------------------
    for i in range(inputData.shape[0]):
        bitStream.write(huffmanTable[int(inputData[i])], bool)

if __name__ == '__main__':
    exit(0)

