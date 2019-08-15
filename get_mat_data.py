'''Data format:
------------

SynthText.zip (size = 42074172 bytes (41GB)) contains 858,750 synthetic
scene-image files (.jpg) split into 200 directories, with 
7,266,866 word-instances, and 28,971,487 characters.

Ground-truth annotations are contained in the file "gt.mat" (Matlab format).
The file "gt.mat" contains the following cell-arrays, each of size 1x858750:

  1. imnames :  names of the image files

  2. wordBB  :  word-level bounding-boxes for each image, represented by
                tensors of size 2x4xNWORDS_i, where:
                   - the first dimension is 2 for x and y respectively,
                   - the second dimension corresponds to the 4 points
                     (clockwise, starting from top-left), and
                   -  the third dimension of size NWORDS_i, corresponds to
                      the number of words in the i_th image.

  3. charBB  : character-level bounding-boxes,
               each represented by a tensor of size 2x4xNCHARS_i
               (format is same as wordBB's above)

  4. txt     : text-strings contained in each image (char array).
               
               Words which belong to the same "instance", i.e.,
               those rendered in the same region with the same font, color,
               distortion etc., are grouped together; the instance
               boundaries are demarcated by the line-feed character (ASCII: 10)

               A "word" is any contiguous substring of non-whitespace
               characters.

               A "character" is defined as any non-whitespace character.'''

'''
这里构建了一些关于.mat文件的函数，诸如读取图片中的信息，利用给定的坐标在图上标出BB，以及根据wordBB切割出word级别的图片。
但是因为某些label本身标注错误，切割图片时会出现图片中word和gtlabel不匹配的情况。这个不好解决，就先不管了。
'''

mat_path = "/home/liming/data/SynthText/gt.mat"
img_root = "/home/liming/data/SynthText/"

import scipy.io as scio

from PIL import Image
import pylab
from pylab import imshow
from pylab import array
from pylab import plot
from pylab import title
import cv2

import numpy as np


data = scio.loadmat(mat_path)
charb = data['charBB']
# charb shape为(1,858750)
# charb[0][x]shape为(2, 4, nchar)，每个字符的4个点的横纵坐标，最后一个参数是字符个数
name = data['imnames']
# name shape为(1,858750)
wordb = data['wordBB']
label = data['txt']

# 需要一个函数，能把一串字符的空格去掉，并以回车符分开
# 是为了去除原始label中的空格和换行
# （原注释中每个的长度都用空格补到了30）
def normalize_str(str):
    str_list = str.split('\n')
    out_list = []
    for i_str in str_list:
        single_str = ''
        for i_char in i_str:
            if i_char != ' ':
                single_str = single_str + i_char
        out_list.append(single_str)
    return out_list

# 输入序号，得到图片中的所有word的label，及其对应的wordBB，以及每个字符的BB
# 图片名字就字符串格式
# label的list应为[label0,label1]
# wordBB应为字典，{label0:[[x1,x2,x3,x4],[y1,y2,y3,y4]],label1..}
# charBB应为字典，{[labal0,char0]:[[x1,x2,x3,x4],[y1,y2,y3,y4]]},[label0,char1]..}
def get_pic_info(num):
    pic_name = data['imnames'][0][num]
    pic_labels = []
    pic_wordBB = {}
    pic_charBB = {}
    word_i = 0
    char_i = 0
    for ilabel in data['txt'][0][num]:
        label_list = normalize_str(ilabel)
        for i_label in label_list:
            pic_labels.append(i_label)
            word_pointer = data['wordBB'][0][num]
            pic_wordBB[i_label] = [[word_pointer[0][0][word_i],word_pointer[0][1][word_i],word_pointer[0][2][word_i],word_pointer[0][3][word_i]], \
                [word_pointer[1][0][word_i],word_pointer[1][1][word_i],word_pointer[1][2][word_i],word_pointer[1][3][word_i]]]
            #print("pic_wordBB[i_label]",pic_wordBB[i_label])
            word_i = word_i +1
            i = 0
            for char in i_label:
                #print("char",char)
                char_pointer = data['charBB'][0][num]
                pic_charBB[i_label,i] = [[char_pointer[0][0][char_i],char_pointer[0][1][char_i],char_pointer[0][2][char_i],char_pointer[0][3][char_i]], \
                    [char_pointer[1][0][char_i],char_pointer[1][1][char_i],char_pointer[1][2][char_i],char_pointer[1][3][char_i]]]
                #print("pic_charBB[i_label,i]",pic_charBB[i_label,i])
                char_i = char_i +1
                i = i + 1
    return pic_name , pic_labels , pic_wordBB , pic_charBB


def drow_point(pic_name , pic_labels , pic_wordBB , pic_charBB):
    path = img_root + pic_name.tostring().decode('utf-8')
    img_path = ''
    for b in path:
        if b != b'\x00'.decode('utf-8'):
            img_path = img_path + b
    # 这里踩了个坑，直接读取得到的文件名是有很多\x00的，而且不仅仅是在前后，所以无法直接用strip去除，也不知道为啥用replace也不能去除

    # 读取图像到数组中
    im = array(Image.open(img_path))
    # 绘制图像
    imshow(im)
    for i_label in pic_labels:
        # 得到word点的横纵坐标
        x = [pic_wordBB[i_label][0][0],pic_wordBB[i_label][0][1],pic_wordBB[i_label][0][2],pic_wordBB[i_label][0][3],pic_wordBB[i_label][0][0]]
        y = [pic_wordBB[i_label][1][0],pic_wordBB[i_label][1][1],pic_wordBB[i_label][1][2],pic_wordBB[i_label][1][3],pic_wordBB[i_label][1][0]]
        plot(x,y,'r')
        i = 0
        for i_char in i_label:
            # 得到char点的横纵坐标
            x = [pic_charBB[i_label,i][0][0],pic_charBB[i_label,i][0][1],pic_charBB[i_label,i][0][2],pic_charBB[i_label,i][0][3],pic_charBB[i_label,i][0][0]]
            y = [pic_charBB[i_label,i][1][0],pic_charBB[i_label,i][1][1],pic_charBB[i_label,i][1][2],pic_charBB[i_label,i][1][3],pic_charBB[i_label,i][1][0]]
            plot(x,y,'b')
            # 得到中心点的横纵坐标
            x_cent = [np.average(x)]
            y_cent = [np.average(y)]
            plot(x_cent,y_cent,'g*')
            i += 1


    # 添加标题，显示绘制的图像
    title(pic_name)
    pylab.show()

# 把完整图片切割成word级别图片
def cut_pic(pic_name , pic_labels , pic_wordBB , pic_charBB):
    rec_pic_name = ''
    for b in pic_name.tostring().decode('utf-8'):
        if b != b'\x00'.decode('utf-8'):
            rec_pic_name = rec_pic_name + b

    img_path = img_root + rec_pic_name
    img = cv2.imread(img_path)
    i = 0
    for i_label in pic_labels:
        cropped = img[int(pic_wordBB[i_label][1][0]):int(pic_wordBB[i_label][1][2]), int(pic_wordBB[i_label][0][0]):int(pic_wordBB[i_label][0][2])]  # 裁剪坐标为[y0:y1, x0:x1]
        store_path = rec_pic_name.split('.')[0].split('/')[1] + '_' + str(i)+ '_' + i_label + '.jpg'
        print(store_path)
        cv2.imwrite(store_path, cropped)
        i += 1

# 查看输出的格式
def see_dtype(pic_name , pic_labels , pic_wordBB , pic_charBB):
    print('dtype pic name:',pic_name.dtype) #  <U18 
    print('dtype pic name:',pic_name.dtype.name) # str576
    print('pic name:',pic_name.tostring()) #bytes
    print('pic name:',pic_name.tostring().decode('utf-8')) # string
    # 我佛了..最开始是numpy的ndarray类型的，所以要先用tostring转换为bytes，再decode才能得到最终的string
    print('dtype pic label:',pic_labels[0].dtype)  # str
    print('dtype pic BB:',pic_wordBB[pic_labels[0]][0][0].dtype) #  float32

pic_name , pic_labels , pic_wordBB , pic_charBB = get_pic_info(1)
cut_pic(pic_name , pic_labels , pic_wordBB , pic_charBB)
drow_point(pic_name , pic_labels , pic_wordBB , pic_charBB)