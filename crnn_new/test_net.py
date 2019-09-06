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

filein = r'/home/liming/code/get_mat/ballet_106_1_0_with.png'
filein2 = r'/home/liming/code/get_mat/ballet_106_1_0_with_mask.png'
width = 100
height = 32
type = 'png'
ResizeImage(filein, 'input.png', width, height, type)
ResizeImage(filein2, 'mask.png', width, height, type)

#os.mkdir('net_result')

loader = transforms.Compose([
    transforms.ToTensor()]) 

unloader = transforms.ToPILImage()

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

input = input.cuda()
mask = mask.cuda()

for i in range (93500,111500,500):
  path = "/home/liming/code/expr/netRES_1_" + str(i) +".pth"
  model.load_state_dict(t.load(path))

  output = model(input)

  output = output.view(1,3,32,100)
  print(criterion(mask,output))
  output = output.squeeze(0)
  output = output.cpu()
  output = unloader(output)
  store_path = 'net_result/test_1'+str(i)+'.jpg'
  output.save(store_path)
