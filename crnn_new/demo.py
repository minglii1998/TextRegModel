import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn

import os
filename = 'test_demo.txt'


model_path = './expr/netCRNN_1_99500.pth'
img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
with open(filename,'a') as f: 
    f.write("\n preds 1:\n")
    f.write(str(preds))
preds = preds.transpose(1, 0).contiguous().view(-1)
with open(filename,'a') as f: 
    f.write("\n preds 2:\n")
    f.write(str(preds))

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
