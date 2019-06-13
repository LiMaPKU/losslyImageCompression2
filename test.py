import torch
import torch.nn.functional as F

downFilter = torch.ones([2,2]).float().unsqueeze(0).unsqueeze(0).cuda()/4
upFilter = torch.ones([2,2]).float().unsqueeze(0).unsqueeze(0).cuda()
x = torch.randint(high=10,size=[1,1,4,4]).float().cuda()

y = F.conv2d(x, downFilter, stride=2)
print(x)
print(y)

y = F.conv_transpose2d(y, upFilter, stride=2)
print(y)

