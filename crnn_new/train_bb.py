from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch import nn
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset_bb as dataset

import models.crnn as crnn
import models.resNet as resNet 

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', help='path to dataset',default="/home/liming/data/ST_lmdb_mask_bb_val_6839/")
parser.add_argument('--valRoot',  help='path to dataset',default="/home/liming/data/ST_lmdb_mask_bb_val_6839/" )
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=5, help='number of epochs to train for')
# TODO(meijieru): epoch -> iter
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=3, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--expr_dir', default='../../expr_mini/', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=100, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
val_dataset = dataset.lmdbDataset(root=opt.valRoot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
# print('length of train loader:',len(train_loader))
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

nclass = 8
nc = 4

criterion = nn.MSELoss()

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


model_mask = resNet.resnext50(num_classes = 3200)
model_bb = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
model_bb.apply(weights_init)
if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    model_bb.load_state_dict(torch.load(opt.pretrained))
print(model_bb)

image_rgb = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgW)
image_2channel = torch.FloatTensor(opt.batchSize, 2, opt.imgH, opt.imgW)
bb = torch.FloatTensor(opt.batchSize, 26, 8)
'''text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)'''


if opt.cuda:
    model_mask.cuda()
    model_bb.cuda()
    image_rgb = image_rgb.cuda()
    image_2channel = image_2channel.cuda()
    bb = bb.cuda()
    criterion = criterion.cuda()
    model_mask = nn.DataParallel(model_mask)
    model_bb = nn.DataParallel(model_bb)

trained_mask_pth = "/home/liming/code/expr_rightloss_2gpu/netRES_3_83500.pth"

model_mask.load_state_dict(torch.load(trained_mask_pth))

image_rgb = Variable(image_rgb)
image_2channel = Variable(image_2channel)
bb = Variable(bb)
'''text = Variable(text)
length = Variable(length)'''

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(model_bb.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(model_bb.parameters())
else:
    optimizer = optim.RMSprop(model_bb.parameters(), lr=opt.lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in model_bb.parameters():
        p.requires_grad = False

    net.eval()

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(val_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_bbs = data
        '''for x in cpu_bbs:
            print(x)
            print(type(x))'''
        batch_size = cpu_images.size(0)
        utils.loadData(image_rgb, cpu_images)
        utils.loadData(bb, cpu_bbs)

        '''t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)'''

        # print('val_image:',image.shape)
        # print('val_mask:',mask.shape)
        mask = model_mask(image_rgb)
        # print('val_preds:',preds.shape)
        mask = mask.view(batch_size,1,32,100)
        #preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        #cost = criterion(preds, text, preds_size, length) / batch_size
        image_2channel = torch.cat((mask,image_rgb),dim=1)

        preds_bb = model_bb(image_2channel)

        # preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        # cost = criterion(preds, text, preds_size, length) / batch_size

        cost = criterion(preds_bb.permute(1,0,2), bb)
        loss_avg.add(cost)

        '''_, preds = preds.max(2)
        preds = preds.squeeze()
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_masks):
            if pred == target.lower():
                n_correct += 1'''

        # 觉得在这种情况下其实是没有所谓的正确的，就设定成loss<0.007吧，没有特殊的原因
        if cost < 0.007 :
            n_correct += 1

    '''raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))'''

    # 下面这个计算方式应该有点问题，因为batch_size不是每一次都是满的，所以这样除肯定会除多
    accuracy = n_correct / float(max_iter * opt.batchSize) 
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_bbs = data
    #print('img: ',cpu_images.shape)
    #print('mask: ',cpu_masks.shape)
    batch_size = cpu_images.size(0)
    utils.loadData(image_rgb, cpu_images)
    utils.loadData(bb, cpu_bbs)
    '''t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)'''

    # print('val_image:',image.shape)
    # print('val_mask:',mask.shape)
    mask = model_mask(image_rgb)
    # print('val_preds:',preds.shape)
    mask = mask.view(batch_size,1,32,100)
    #preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    #cost = criterion(preds, text, preds_size, length) / batch_size
    image_2channel = torch.cat((mask,image_rgb),dim=1)

    preds_bb = model_bb(image_2channel)

    #print('preds_bb shape:',preds_bb.shape)
    #print('bb shape:',bb.shape)
    cost = criterion(preds_bb.permute(1,0,2), bb)
    model_bb.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    i = 0
    
    while i < len(train_loader):
        for p in model_bb.parameters():
            p.requires_grad = True
        model_bb.train()

        cost = trainBatch(model_bb, criterion, optimizer)
        #print(i)
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            #pass
            val(model_bb,val_dataset, criterion)

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(
                model_bb.state_dict(), '{0}/netRES_{1}_{2}.pth'.format(opt.expr_dir, epoch, i))
