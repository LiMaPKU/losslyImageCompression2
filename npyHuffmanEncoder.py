import numpy

inputData = numpy.load('./output/encData.npy').squeeze()
minV = inputData.min()
inputData = inputData - minV
print(inputData.shape,inputData.min(),inputData.max())
for i in range(inputData.shape[0]):
    print('--------',i,'--------')
    for j in range(inputData.shape[1]):
        print(inputData[i][j])