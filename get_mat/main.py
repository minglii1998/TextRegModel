# used to try and test model
from get_mat_data import get_pic_info
from get_mat_data import cut_pic_old
from get_mat_data import drow_point
from get_mat_data import cut_pic
from get_mat_data import get_gt_mask

import scipy.io as scio
import os

from torchvision import transforms
import numpy as np
import torch as t
from PIL import Image

mat_path = "/home/liming/data/SynthText/gt.mat"
img_root = "/home/liming/data/SynthText/"
store_prepared_path = "./"

'''if os.path.exists(store_prepared_path):
    pass
else:
    os.mkdir(store_prepared_path)'''

data = scio.loadmat(mat_path)

pic_name , pic_labels , pic_wordBB , pic_charBB = get_pic_info(data,1)
'''print (pic_charBB)
with open('pic_charbb.txt','w') as f:
  f.write(str(pic_charBB))'''
path_list, label_list, mask_path_last, bb_list = cut_pic_old(pic_name , pic_labels , pic_wordBB , pic_charBB, store_prepared_path, img_root)
# print(bb_list[0])
bb_real_list =eval(bb_list[0])
bb_all = []
word_len = len(bb_real_list)
print(word_len)
char_count = 0
for num in range(20):
  bb8 = []
  if num < word_len: 
    bb8.append(bb_real_list[num][0][0])
    bb8.append(bb_real_list[num][1][0])
    bb8.append(bb_real_list[num][0][1])
    bb8.append(bb_real_list[num][1][1])
    bb8.append(bb_real_list[num][0][2])
    bb8.append(bb_real_list[num][1][2])
    bb8.append(bb_real_list[num][0][3])
    bb8.append(bb_real_list[num][1][3])
    bb_all.append(bb8)
  else:
    bb_all.append([-1,-1,-1,-1,-1,-1,-1,-1])
  
print(bb_all)
print(label_list[0])

'''def ResizeImage(filein, fileout, width, height, type):
  img = Image.open(filein)
  out = img.resize((width, height),Image.BILINEAR) #resize image with high-quality
  out.save(fileout, type)

if __name__ == "__main__":
  filein = r'/home/liming/code/get_mat/ballet_106_1_0_with_mask.png'
  fileout = r'testout.png'
  width = 100
  height = 32
  type = 'png'
  ResizeImage(filein, fileout, width, height, type)'''

#drow_point(pic_name , pic_labels , pic_wordBB , pic_charBB, img_root)
'''loader = transforms.Compose([
    transforms.ToTensor()]) 

unloader = transforms.ToPILImage()

img = Image.open('/home/liming/code/get_mat/testout.png').convert('RGB')

image = loader(img).unsqueeze(0)
image = image.to(t.float)
print(image.shape)
img = image[0][2]
print(img[20][60])
print(img.shape)
img = unloader(img)
img.show()'''