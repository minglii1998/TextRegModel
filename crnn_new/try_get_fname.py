import scipy.io as sio
import numpy as np
import glob
import os
import cv2
from cv2 import cv2 as cv
import h5py

import utils
import torch
from torch.autograd import Variable
import lmdb
import models.crnn as crnn
import six
import sys
from PIL import Image
import matplotlib.pyplot as plt

root = "/home/liming/data/ST_lmdb_mask_bb_val_6839/"
root1 = "/mnt/data_share/text_recog/data_lmdb_release/evaluation/IC03_860/"
#test_folder = 'testfolder'
#os.mkdir(test_folder)


'''# 遍历文件夹下的所有目录

test_path = r"/home/liming/code/expr_rightloss_2gpu/"
g = os.walk(test_path)
test_list = []
label_list = []

for path,dir_list,file_list in g:  
    new_list = glob.glob(os.path.join(path, '*.pth'))
    test_list = test_list + new_list

for item in test_list:
    label_list.append(int(item.split('/')[-1].split('.')[0].split('_')[2]))
    
print(max(label_list))'''



# 尝试使用lmdb
env = lmdb.open(
    root,
    max_readers=1,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False)

with env.begin(write=False) as txn:
    txn.cursor()

for i in range(100,101):

    Index = i+1
    mask_key = 'mask-%09d' % Index
    label_key = 'label-%09d' % Index
    bb_key = 'bb-%09d' % Index

    with env.begin(write=False) as txn:
        label = txn.get(label_key.encode()).decode()
        bb = txn.get(bb_key.encode()).decode()

        imgbuf = txn.get(mask_key.encode())
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        image = Image.open(buf).convert('L')
        print(label)
        print(len(eval(bb)))
        print(eval(bb))
    image.save(label+'.jpg')


'''
root = '/mnt/data_share/text_recog/data_lmdb_release/evaluation/IC03_860'
model_path = './expr/netCRNN_1_99500.pth'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
pic_count = 0
right_count = 0

env = lmdb.open(
    root,
    max_readers=1,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False)

Index = 2
img_key = 'image-%09d' % Index
label_key = 'label-%09d' % Index

wrong_folder = '../'
wrong_folder = wrong_folder + 'wrong:' + model_path.split('/')[2].split('.')[0] + '+' + root.split('/')[6]
os.mkdir(wrong_folder)

with env.begin(write=False) as txn:
    label_key = 'label-%09d' % Index
    label = txn.get(label_key.encode()).decode()

    imgbuf = txn.get(img_key.encode())
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    image = Image.open(buf).convert('L')

wrong_img_name = wrong_folder + '/' + str(img_key) + '_' + str(label) + '.jpg'
# wrong_img_name = wrong_folder + '/' +  '1111.jpg'
image.save(wrong_img_name)
'''

'''
model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

env = lmdb.open(
    root,
    max_readers=1,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False)

with env.begin(write=False) as txn:
    nSamples = int(txn.get('num-samples'.encode()))
    print(nSamples)

for Index in range(nSamples):

    pic_count += 1

    img_key = 'image-%09d' % Index
    label_key = 'label-%09d' % Index

    with env.begin(write=False) as txn:
        label_key = 'label-%09d' % Index
        label = txn.get(label_key.encode()).decode()

        imgbuf = txn.get(img_key.encode())
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        image = Image.open(buf).convert('L')

    transformer = dataset.resizeNormalize((100, 32))
    image = transformer(image)

    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('%-20s => %-20s' % (raw_pred, sim_pred))
    str_pred = "".join(sim_pred)

    if (str_pred == label):
        right_count+=1
    else:
        print('predicted: %-20s,real: %-20s,key: %s ' 
        % (sim_pred, label, img_key))


print ("number of tested: %d" % pic_count)
print ("number of right: %d" % right_count)
print ("Accuracy : %f" % (right_count/pic_count))
'''



# 获得lmdb中的数据
'''
test_dataset = dataset.lmdbDataset(
    root='/mnt/data_share/text_recog/data_lmdb_release/evaluation/IC03_860', transform=dataset.resizeNormalize((100, 32)))

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
image = Variable(image)
text = Variable(text)

data_loader = torch.utils.data.DataLoader(
    test_dataset, shuffle=True)
val_iter = iter(data_loader)

n_correct = 0
n_count = 0
converter = utils.strLabelConverter(opt.alphabet)

for i in range(len(data_loader)):
    n_count += 1
    data = val_iter.next()

    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, _ = converter.encode(cpu_texts)
    utils.loadData(text, t)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

    _, preds = preds.max(2)
    preds = preds.squeeze()
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    for pred, target in zip(sim_preds, cpu_texts):
        if pred == target.lower():
            n_correct += 1
        else:
            print('predicted: %-20s,real: %-20s '  % (pred, target))

print ("number of tested: %d" % n_count)
print ("number of right: %d" % n_correct)
print ("Accuracy : %f" % (n_correct/n_count))
'''


# 直接读文件中的图片的
'''
test_path = '/home/liming/data/MJSynth/ramdisk/max/90kDICT32px/1/1/'

test_list = glob.glob(os.path.join(test_path, '*.jpg'))
label_list = []

for item in test_list:
    label_list.append(item.split('_')[1].lower())
    
print (label_list)
'''

# 下面这些是之前用来读.mat文件的，现在先不管了
'''
data=h5py.File(test_label_path)
k = list(data.keys())
ds = data['digitStruct']
print(list(ds))
#print(list(ds['bbox']))
print(list(data[ds['bbox'][0,0]]))

#print(list(data[data[ds['bbox'][0,0]][0,0]]))



test = data['digitStruct/name']
st = test[0][0]
obj = data[st]
print(list(obj))


struArray = data['digitStruct']

print(struArray)

print(data.values())

print(data['digitStruct'].values())
'''
