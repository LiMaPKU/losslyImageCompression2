from PIL import Image
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim
import sys
import os
import pytorch_gdn
import pytorch_msssim

img1 = Image.open('./output/input.bmp').convert('L')
img1.save('./output/jpg.jpg', quality=6)
img2 = Image.open('./output/jpg.jpg')
img1 = torch.from_numpy(numpy.asarray(img1).astype(float).reshape([1, 1, 256, 256])).float().cuda()
img2 = torch.from_numpy(numpy.asarray(img2).astype(float).reshape([1, 1, 256, 256])).float().cuda()
MS_SSIM = pytorch_msssim.ms_ssim(img1, img2, data_range=255, size_average=True)
print(MS_SSIM)