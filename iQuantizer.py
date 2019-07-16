#智能量化器
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F


class Quantize(torch.autograd.Function): # 输入为b*c*m*n 例如n*64*16*16
    @staticmethod
    def forward(ctx, input, qLevel):
        shapeI = input.shape
        input = input.view(shapeI[0],-1) # 将输入从b*c*m*n变成b*(c*m*n)

        for i in range(qLevel):
            sfI = F.softmax(input - i)


        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output

def quantize(input):
    return Quantize.apply(input)


