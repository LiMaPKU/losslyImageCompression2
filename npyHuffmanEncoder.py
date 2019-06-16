import numpy
from scipy import stats
import torch
import huffmanEncodeFunction
from bitstream import BitStream
# 不使用科学计数法输出 不输出省略号 浮点输出2位小数
numpy.set_printoptions(suppress=True, threshold=numpy.inf, precision=2)
# z扫描顺序
bitStream = BitStream() # 比特流
# 读取数据
inputData = numpy.load('./output/encData.npy').squeeze()
minV = inputData.min()
maxV = inputData.max()
print('原始数据的最小值最大值分别为',minV,maxV)
inputData = inputData - int((minV + maxV)/2) # 将数据中心化
newMinV = inputData.min()
newMaxV = inputData.max()
print('中心化后的最小值最大值分别为',newMinV,newMaxV)


# for i in range(inputData.shape[0]):
#     print('--------',i,'--------')
#     print(inputData[i])


modeList = numpy.zeros(shape=[inputData.shape[0]], dtype=int) # 保存每个通道的众数
dModeList = numpy.zeros(shape=[inputData.shape[0]], dtype=int) # 保存众数的前向差分
for i in range(inputData.shape[0]):
    # print('--------',i,'--------')
    # 在每个通道内，[0][0]位置元素保存众数，其他元素均减去[0][0]

    modeV = int(stats.mode(inputData[i].flatten())[0][0])
    inputData[i] = inputData[i] - modeV
    inputData[i][0][0] = modeV
    modeList[i] = modeV
    if(i==0):
        dModeList[i] = modeList[i]
    else:
        dModeList[i] = modeList[i] - modeList[i-1]


    # print(inputData[i])

newMinV = inputData.min()
newMaxV = inputData.max()
print('每个通道减去众数后，最小值最大值为',newMinV,newMaxV)
inputDataHistc = torch.histc(torch.from_numpy(inputData).cuda(),min = newMinV,max=newMaxV,bins=int(inputData.max()-inputData.min()+1)).cpu().numpy()
print('数据分布图为',inputDataHistc)
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
print('各个通道的众数',modeList)
print('众数的前向差分',dModeList)
huffmanEncodeFunction.encodeDModeList(bitStream, dModeList)

# 对数据进行截断
newMinV = max8startPos
newMaxV = newMinV + 7
inputData[inputData<newMinV] = newMinV
inputData[inputData>newMaxV] = newMaxV
newMinVOffset = newMinV + 3 # 偏移量
inputData = inputData - newMinVOffset # 将数据全部转换到[-3, 4]
# 截断后数据都处于[-3, 4]
print('对数据进行截断、平移到[-3,4]')



