from PIL import Image
import numpy
class datasetReader:

    imgDir = '/datasets/MLG/wfang/imgCompress/256bmp'
    imgSumInMemory = 0 # 提前从硬盘读取，保存在内存中的图片数量
    trainedImgNum = 0 # 已经被读取进入gpu的图片数量
    imgData = numpy.array([0])

    imgSum = 220000 # 一共有22万张图片
    readSeq = numpy.array([0]) # 要读取的图片标号


    def __init__(self, batchSize):
        numpy.random.seed()
        self.imgSumInMemory = 8 * batchSize

        self.imgData = numpy.empty([self.imgSumInMemory, 1, 256, 256])
        self.readImgToMemory()


    def readImgToMemory(self):
        self.readSeq = numpy.random.randint(low=0, high=self.imgSum, size=[self.imgSumInMemory])
        for i in range(self.imgSumInMemory):
            img = Image.open('/datasets/MLG/wfang/imgCompress/256bmp/' + str(self.readSeq[i]) + '.bmp').convert('L')
            self.imgData[i] = numpy.asarray(img).astype(float).reshape([1, 256, 256])

    def readImg(self):
        retData = self.imgData[self.trainedImgNum]
        self.trainedImgNum = self.trainedImgNum + 1
        if(self.trainedImgNum == self.imgSumInMemory):
            self.readImgToMemory()
            self.trainedImgNum = 0
        return retData











