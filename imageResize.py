from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

imgDirList = ['/datasets/MLG/voc2007/VOCdevkit/VOC2007/JPEGImages','/datasets/MLG/VOC2012/JPEGImages',
          '/datasets/MLG/coco/coco2014/train2014','/datasets/MLG/coco/train2017']

num = 0
for imgDir in imgDirList:
    imgList = os.listdir(imgDir)
    for fileName in imgList:
        img = Image.open(imgDir + '/' + fileName)
        if(img.width>=256 and img.height>=256):
            img = img.crop((0, 0, 256, 256))
            img.save('/datasets/MLG/wfang/imgCompress/256bmp/' + str(num) + '.bmp')
            num = num + 1
            if(num%100==0):
                print(num)