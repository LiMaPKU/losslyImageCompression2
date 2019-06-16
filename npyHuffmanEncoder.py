import numpy
from scipy import stats
import torch
import huffmanEncodeFunction
from bitstream import BitStream
import huffmanTable
# 不使用科学计数法输出 不输出省略号 浮点输出2位小数
numpy.set_printoptions(suppress=True, threshold=numpy.inf, precision=2)
# z扫描顺序
bitStream = BitStream() # 比特流
# 读取数据
inputData = numpy.load('./output/encData.npy').squeeze()
minV = inputData.min()
maxV = inputData.max()
print('原始数据的最小值最大值分别为',minV,maxV)
#inputData = inputData - int((minV + maxV)/2) # 将数据中心化
#newMinV = inputData.min()
#newMaxV = inputData.max()
#print('中心化后的最小值最大值分别为',newMinV,newMaxV)


# for i in range(inputData.shape[0]):
#     print('--------',i,'--------')
#     print(inputData[i])


modeList = numpy.zeros(shape=[inputData.shape[0]], dtype=int) # 保存每个通道的众数
dModeList = numpy.zeros(shape=[inputData.shape[0]], dtype=int) # 保存众数的前向差分
for i in range(inputData.shape[0]):
    # print('--------',i,'--------')

    modeV = int(stats.mode(inputData[i].flatten())[0][0])
    inputData[i] = inputData[i] - modeV
    modeList[i] = modeV
    if(i==0):
        dModeList[i] = modeList[i]
    else:
        dModeList[i] = modeList[i] - modeList[i-1]


    # print(inputData[i])

print('各个通道的众数',modeList)
print('众数的前向差分',dModeList)
huffmanEncodeFunction.huffmanEncode(bitStream, dModeList)
zCode = []
for i in range(inputData.shape[0]):
    zCode.extend((inputData[i].flatten()[huffmanTable.zigzagOrder]).tolist())

huffmanEncodeFunction.runValueHuffmanEncode(bitStream, numpy.asarray(zCode))
outputFile = open('./output/outputFile.b', 'wb+')
# write encoded data
bitLength = bitStream.__len__()
filledNum = 8 - bitLength % 8
if(filledNum!=0):
    bitStream.write(numpy.ones([filledNum]).tolist(),bool) # 补全为字节（b的数量应该是8整数倍）
sosBytes = bitStream.read(bytes)
for i in range(len(sosBytes)):
    outputFile.write(bytes([sosBytes[i]]))
    if(sosBytes[i]==255):
        outputFile.write(bytes([0])) # FF to FF 00


outputFile.close()

'''
max8startPos = 0
max8 = 0
for i in range(inputDataHistc.shape[0]-7):
    sum8 = numpy.sum(inputDataHistc[i:i+8]) # [i,i+7]
    print('[',i+newMinV,i+newMinV+7,']区间总数',sum8)
    if(sum8>=max8):
        max8 = sum8
        max8startPos = i
max8startPos = max8startPos + newMinV
print('分布最为集中的区间是[',max8startPos,max8startPos+7,']')
# 对数据进行截断
newMinV = max8startPos
newMaxV = newMinV + 7
inputData[inputData<newMinV] = newMinV
inputData[inputData>newMaxV] = newMaxV
newMinVOffset = newMinV + 3 # 偏移量
inputData = inputData - newMinVOffset # 将数据全部转换到[-3, 4]
# 截断后数据都处于[-3, 4]
print('对数据进行截断、平移到[-3,4]')
'''




