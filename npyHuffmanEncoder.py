import numpy
import torch
import huffmanEncodeFunction
from bitstream import BitStream
zigzagOrder = numpy.array([0,1,16,32,17,2,3,18,33,48,64,49,34,19,4,5,20,35,50,65,80,96,81,66,51,36,21,6,7,22,37,52,67,82,97,112,128,113,98,83,68,53,38,23,8,9,24,39,54,69,84,99,114,129,144,160,145,130,115,100,85,70,55,40,25,10,11,26,41,56,71,86,101,116,131,146,161,176,192,177,162,147,132,117,102,87,72,57,42,27,12,13,28,43,58,73,88,103,118,133,148,163,178,193,208,224,209,194,179,164,149,134,119,104,89,74,59,44,29,14,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,241,226,211,196,181,166,151,136,121,106,91,76,61,46,31,47,62,77,92,107,122,137,152,167,182,197,212,227,242,243,228,213,198,183,168,153,138,123,108,93,78,63,79,94,109,124,139,154,169,184,199,214,229,244,245,230,215,200,185,170,155,140,125,110,95,111,126,141,156,171,186,201,216,231,246,247,232,217,202,187,172,157,142,127,143,158,173,188,203,218,233,248,249,234,219,204,189,174,159,175,190,205,220,235,250,251,236,221,206,191,207,222,237,252,253,238,223,239,254,255])

# 不使用科学计数法输出 不输出省略号 浮点输出2位小数
numpy.set_printoptions(suppress=True, threshold=numpy.inf, precision=2)
bitStream = BitStream() # 比特流
# 读取数据
inputData = numpy.load('./output/encData.npy').squeeze()
minV = inputData.min()
maxV = inputData.max()
print('原始数据的最小值最大值分别为',minV,maxV)


meanList = numpy.zeros(shape=[inputData.shape[0]], dtype=int)  # 保存每个通道的均值
for i in range(inputData.shape[0]):
    # print('--------',i,'--------')

    meanV = int(inputData[i].mean())
    inputData[i] = inputData[i] - meanV
    meanList[i] = meanV

print('各个通道的均值', meanList)

zCode = []
for i in range(inputData.shape[0]):
    zCode.extend((inputData[i].flatten()[zigzagOrder]).tolist())
zCode = numpy.asarray(zCode)

bitStream = BitStream()
huffmanEncodeFunction.valueHuffmanEncode(zCode, bitStream)







outputFile = open('./output/outputFile.b', 'wb+')
# write encoded data
bitLength = bitStream.__len__()
filledNum = 8 - bitLength % 8
if(filledNum!=0):
    bitStream.write(numpy.ones([filledNum]).tolist(),bool) # 补全为字节（b的数量应该是8整数倍）
sosBytes = bitStream.read(bytes)
for i in range(len(sosBytes)):
    outputFile.write(bytes([sosBytes[i]]))
    #if(sosBytes[i]==255):
        #outputFile.write(bytes([0])) # FF to FF 00


outputFile.close()






