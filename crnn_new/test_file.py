'''
这个文件是用来测试同一个图输入到不同时期的网络中，能够可视化的结果
懒得改成泛用性更高的代码了...本来想加入parser的还是算了...懒
'''

import numpy as np
import torch as t

from PIL import Image
from torchvision import transforms
import torch.optim as optim
from torch import nn

import models.resNet as resNet 
import os

def ResizeImage(filein, fileout, width, height, type):
  img = Image.open(filein)
  out = img.resize((width, height),Image.BILINEAR) #resize image with high-quality
  out.save(fileout, type)

loader = transforms.Compose([
    transforms.ToTensor()]) 
unloader = transforms.ToPILImage()

# 这里是为了获得测试用的图片和mask，基本户是哪个用一次就够了
'''filein = r'/home/liming/code/get_mat/ballet_106_1_0_with.png'
filein2 = r'/home/liming/code/get_mat/ballet_106_1_0_with_mask.png'
width = 100
height = 32
type = 'png'
ResizeImage(filein, 'input.png', width, height, type)
ResizeImage(filein2, 'mask.png', width, height, type)'''

# 最终将图片保存至哪里
result_path = '../net_result3d_64bs'

if not os.path.exists(result_path):
  os.mkdir(result_path)


criterion = nn.MSELoss()
model = resNet.resnext50(num_classes = 9600)

model = model.cuda()
model = nn.DataParallel(model)

image = Image.open('input.png').convert('RGB')
image = loader(image).unsqueeze(0)
input = image.to(t.float)
image = Image.open('mask.png').convert('RGB')
image = loader(image).unsqueeze(0)
mask = image.to(t.float)
# mask = mask[0][2]
# mask = mask[:,2,:,:]
# print(mask.size())

input = input.cuda()
mask = mask.cuda()

for j in range(0,1):
  for i in range(0,99501,500):

    try:
      load_path = '/home/liming/code/expr_bs64_3d/netRES_' + str(j) + '_' + str(i) + '.pth'
      model.load_state_dict(t.load(load_path))
    except FileNotFoundError:
      print(str(i) + ': No File')
      continue

    model.eval()
    output = model(input)

    output = output.view(3,32,100)
    loss = criterion(mask,output)
    loss = loss.data

    if loss > 2:
      os.remove(load_path)
      print(str(i) + ': Path delete')
      continue
    elif loss > 1:
      print(str(i) + ': Pass')
      continue
    else:
      print(str(i) + ': Print')
      loss = loss.cpu()
      loss = loss.numpy()

      output = output.squeeze(0)
      output = output.cpu()
      output = unloader(output)
      store_path = result_path + '/' + str(j) + '_' + str(i) + '_' + str(loss) + '.jpg'
      output.save(store_path)
