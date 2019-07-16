import torch
import torch.nn.functional as F
class edgeMSE(torch.nn.Module):# 带有边缘权重的MSE
    def __init__(self):
        super(edgeMSE, self).__init__()

    def forward(ctx, origin, rebuild):


        edgeFilter = torch.tensor([
        [-0.2357, -0.2774, -0.3162, -0.3333, -0.3162, -0.2774, -0.2357],
        [-0.2774, -0.3536, -0.4472, -0.5000, -0.4472, -0.3536, -0.2774],
        [-0.3162, -0.4472, -0.7071, -1.0000, -0.7071, -0.4472, -0.3162],
        [-0.3333, -0.5000, -1.0000, 20.8451, -1.0000, -0.5000, -0.3333],
        [-0.3162, -0.4472, -0.7071, -1.0000, -0.7071, -0.4472, -0.3162],
        [-0.2774, -0.3536, -0.4472, -0.5000, -0.4472, -0.3536, -0.2774],
        [-0.2357, -0.2774, -0.3162, -0.3333, -0.3162, -0.2774, -0.2357]]).float().unsqueeze(0).unsqueeze(0).cuda()

        mse = torch.pow(origin - rebuild, 2)
        weight = torch.abs(F.conv2d(torch.nn.functional.pad(origin,(3,3,3,3),'replicate'), edgeFilter/20.8451))/255
        wMse = torch.mean(mse*weight)
        return wMse

EdgeMSELoss = edgeMSE()

class edgeMCE(torch.nn.Module):# 带有边缘权重的MCE 比edgeMSE更重视边缘处的损失
    def __init__(self):
        super(edgeMCE, self).__init__()

    def forward(ctx, origin, rebuild):


        edgeFilter = torch.tensor([
        [-0.2357, -0.2774, -0.3162, -0.3333, -0.3162, -0.2774, -0.2357],
        [-0.2774, -0.3536, -0.4472, -0.5000, -0.4472, -0.3536, -0.2774],
        [-0.3162, -0.4472, -0.7071, -1.0000, -0.7071, -0.4472, -0.3162],
        [-0.3333, -0.5000, -1.0000, 20.8451, -1.0000, -0.5000, -0.3333],
        [-0.3162, -0.4472, -0.7071, -1.0000, -0.7071, -0.4472, -0.3162],
        [-0.2774, -0.3536, -0.4472, -0.5000, -0.4472, -0.3536, -0.2774],
        [-0.2357, -0.2774, -0.3162, -0.3333, -0.3162, -0.2774, -0.2357]]).float().unsqueeze(0).unsqueeze(0).cuda()

        mce = torch.abs(torch.pow(origin - rebuild, 3))
        weight = torch.abs(F.conv2d(torch.nn.functional.pad(origin,(3,3,3,3),'replicate'), edgeFilter/20.8451))/255
        wMce = torch.mean(mce*weight)
        return wMce

EdgeMCELoss = edgeMCE()

if __name__ == '__main__':  # 如果运行本py文件 就运行main函数
    from PIL import Image
    import numpy
    import math
    '''
        edgeFilter = torch.zeros(size=[7, 7], dtype=torch.float)
    for i in range(7):
        for j in range(7):
            if(i!=3 or j!=3):
                edgeFilter[i][j] = -1/math.sqrt(pow(i-3,2) + pow(j-3,2))
    print(edgeFilter)
    print(edgeFilter.sum())
    exit(0)
    '''

    img = Image.open('./input2.bmp').convert('L')
    img = numpy.asarray(img).astype(float).reshape([1, 1, 256, 256])
    edgeFilter = torch.tensor([
        [-0.2357, -0.2774, -0.3162, -0.3333, -0.3162, -0.2774, -0.2357],
        [-0.2774, -0.3536, -0.4472, -0.5000, -0.4472, -0.3536, -0.2774],
        [-0.3162, -0.4472, -0.7071, -1.0000, -0.7071, -0.4472, -0.3162],
        [-0.3333, -0.5000, -1.0000, 20.8451, -1.0000, -0.5000, -0.3333],
        [-0.3162, -0.4472, -0.7071, -1.0000, -0.7071, -0.4472, -0.3162],
        [-0.2774, -0.3536, -0.4472, -0.5000, -0.4472, -0.3536, -0.2774],
        [-0.2357, -0.2774, -0.3162, -0.3333, -0.3162, -0.2774, -0.2357]]).float().unsqueeze(0).unsqueeze(0).cuda()
    img = torch.from_numpy(img).float().cuda()
    img = torch.nn.functional.pad(img,(3,3,3,3),'replicate')

    edgeImg = torch.abs(F.conv2d(img, edgeFilter))
    print(edgeImg.shape)
    edgeImg = edgeImg.cpu().detach().numpy().astype(int).reshape([256, 256])
    edgeImg = Image.fromarray(edgeImg.astype('uint8')).convert('L')
    edgeImg.save('./output.bmp')



