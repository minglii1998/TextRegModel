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

#mat_path = "/home/liming/data/SynthText/gt.mat"
#img_root = "/home/liming/data/SynthText/"
#store_prepared_path = "/home/liming/data/prepare_lmdb"

import scipy.io as scio

from PIL import Image
import pylab
from pylab import imshow
from pylab import array
from pylab import plot
from pylab import title
import cv2

import numpy as np
import math

import os

#os.mkdir(store_prepared_path)
#data = scio.loadmat(mat_path)
'''
charb = data['charBB']
# charb shape为(1,858750)
# charb[0][x]shape为(2, 4, nchar)，每个字符的4个点的横纵坐标，最后一个参数是字符个数
name = data['imnames']
# name shape为(1,858750)
wordb = data['wordBB']
label = data['txt']'''

# 需要一个函数，能把一串字符的空格去掉，并以回车符分开
# 是为了去除原始label中的空格和换行
# （原注释中每个的长度都用空格补到了30）
def normalize_str_old(str):
    str_list = str.split('\n')
    out_list = []
    for i_str in str_list:
        single_str = ''
        for i_char in i_str:
            if i_char != ' ':
                single_str = single_str + i_char
        out_list.append(single_str)
    return out_list

# 有些字符串是以空格分隔的，old方法没考虑到这个
def normalize_str(str):
    str = str.strip()
    str_list = str.split('\n')
    out_list = []
    temp_list = []
    for i_str in str_list:
        i_str = i_str.strip()
        temp_list = i_str.split(' ')
        out_list += temp_list
    return out_list

# 在不正常的额数据中（如398），data['wordBB'][0][num]的值是
# 正常的是(2,4,n)
def debug_wordBB(word_pointer, word_i, data, num):
    print('name:',data['imnames'][0][num])
    word_pointer = data['wordBB'][0][num]
    print('dim:',data['wordBB'][0][num].ndim)
    print('content:',data['wordBB'][0][num])
    print('word_pointer[0][0][word_i]:',word_pointer[0][0][word_i])
    print('word_pointer[0][1][word_i]:',word_pointer[0][1][word_i])
    print('word_pointer[0][2][word_i]:',word_pointer[0][2][word_i])
    print('word_pointer[0][3][word_i]:',word_pointer[0][3][word_i])
    print('word_pointer[1][0][word_i]:',word_pointer[1][0][word_i])
    print('word_pointer[1][1][word_i]:',word_pointer[1][1][word_i])
    print('word_pointer[1][2][word_i]:',word_pointer[1][2][word_i])
    print('word_pointer[1][3][word_i]:',word_pointer[1][3][word_i])

# 输入序号，得到图片中的所有word的label，及其对应的wordBB，以及每个字符的BB
# 图片名字就字符串格式
# label的list应为[label0,label1]
# wordBB应为字典，{label0:[[x1,x2,x3,x4],[y1,y2,y3,y4]],label1..}
# charBB应为字典，{[labal0,0]:[[x1,x2,x3,x4],[y1,y2,y3,y4]]},[label0,1]..}
def get_pic_info(data,num):
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
            # 之前默认所有的图片的wordBB都是三维的，然而若只有一个标注，实际标注是二维的。
            if word_pointer.ndim == 2:
                continue
            try :
                pic_wordBB[i_label] = [[word_pointer[0][0][word_i],word_pointer[0][1][word_i],word_pointer[0][2][word_i],word_pointer[0][3][word_i]], \
                    [word_pointer[1][0][word_i],word_pointer[1][1][word_i],word_pointer[1][2][word_i],word_pointer[1][3][word_i]]]
            except IndexError:
                continue
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
    
    '''
    print('pic_name:',pic_name)
    print('pic_labels:',pic_labels)
    print('pic_wordBB:',pic_wordBB)
    print('pic_charBB:',pic_charBB)
    '''
    return pic_name , pic_labels , pic_wordBB , pic_charBB


def drow_point(pic_name , pic_labels , pic_wordBB , pic_charBB, img_root):
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
# 顺便返回所有图片的路径和label
# 这里需要改成利用left right up down来切割，这样就不会出现切割错误
def cut_pic_old(pic_name , pic_labels , pic_wordBB , pic_charBB, prefix_store_path, img_root):
    if len(pic_labels) == 1:
        return [],[],[]
    pic_path_list = []
    pic_label_list = []
    mask_path_list = []
    bb_list_temp = []
    bb_list = []
    rec_pic_name = ''
    for b in pic_name.tostring().decode('utf-8'):
        if b != b'\x00'.decode('utf-8'):
            rec_pic_name = rec_pic_name + b

    img_path = img_root + rec_pic_name
    img = cv2.imread(img_path)
    i = 0
    for i_label in pic_labels:

        try:
            left = min(int(pic_wordBB[i_label][0][0]),int(pic_wordBB[i_label][0][1]),int(pic_wordBB[i_label][0][2]),int(pic_wordBB[i_label][0][3]))
            left = max(left,0)
            right = max(int(pic_wordBB[i_label][0][0]),int(pic_wordBB[i_label][0][1]),int(pic_wordBB[i_label][0][2]),int(pic_wordBB[i_label][0][3]))+5
            up = min(int(pic_wordBB[i_label][1][0]),int(pic_wordBB[i_label][1][1]),int(pic_wordBB[i_label][1][2]),int(pic_wordBB[i_label][1][3]))-5
            up = max(up,0)
            down = max(int(pic_wordBB[i_label][1][0]),int(pic_wordBB[i_label][1][1]),int(pic_wordBB[i_label][1][2]),int(pic_wordBB[i_label][1][3]))+5
            cropped = img[up:down,left:right]
            #cropped = img[int(pic_wordBB[i_label][1][0]):int(pic_wordBB[i_label][1][2]), int(pic_wordBB[i_label][0][0]):int(pic_wordBB[i_label][0][2])]  # 裁剪坐标为[y0:y1, x0:x1]
        except KeyError:
            continue
        #print(left,right,up,down)

        i_char = 0
        for _ in i_label:
            get_all_cropped_point(up,left,pic_charBB[i_label,i_char])
            bb_list_temp.append(pic_charBB[i_label,i_char])
            i_char += 1 

        try :
            if not cropped_point_verify(pic_charBB[i_label,0]):
                continue
        except KeyError:
            continue     

        bb_list.append(str(bb_list_temp))
        store_path = rec_pic_name.split('.')[0].split('/')[1] + '_' + str(i)+ '_' + i_label + '.png'
        store_mask_path = rec_pic_name.split('.')[0].split('/')[1] + '_' + str(i)+ '_' + i_label + '_mask.png'
        store_path = prefix_store_path + '/' + store_path
        store_mask_path = prefix_store_path + '/' + store_mask_path
        #print(store_path)
        pic_path_list.append(store_path)
        pic_label_list.append(i_label)
        mask_path_list.append(store_mask_path)
        cv2.imwrite(store_path, cropped)

        # print(cropped.size)
        get_gt_mask(cropped,pic_wordBB,pic_charBB,i_label,store_mask_path)
        # debug_charBB_rotated(store_path,pic_charBB,i_label)

        i += 1
    return pic_path_list, pic_label_list, mask_path_list,bb_list

# 用来得到旋转的角度
def get_angle(x1,y1,x3,y3,x4,y4):
    alpha = math.atan((y4-y3)/(x4-x3))*180/math.pi
    return alpha

def get_one_rotated_point(c1,c2,x,y,angle):
    angle = -angle
    '''z = math.sqrt((x-c1)**2+(y-c2)**2)
    beta = math.atan((y-c2)/(x-c1))
    theta = beta + angle/180*math.pi
    x = c1 + z * math.cos(theta)
    y = c2 + z * math.sin(theta)'''
    x = (x-c1)*math.cos(angle/180*math.pi)-(y-c2)*math.sin(angle/180*math.pi)+c1
    y = (x-c1)*math.sin(angle/180*math.pi)+(y-c2)*math.cos(angle/180*math.pi)+c2
    return x,y

def get_all_rotated_point(c1,c2,BB,angle):
    BB[0][0], BB[1][0] = get_one_rotated_point(c1,c2,BB[0][0],BB[1][0],angle)
    BB[0][1], BB[1][1] = get_one_rotated_point(c1,c2,BB[0][1],BB[1][1],angle)
    BB[0][2], BB[1][2] = get_one_rotated_point(c1,c2,BB[0][2],BB[1][2],angle)
    BB[0][3], BB[1][3] = get_one_rotated_point(c1,c2,BB[0][3],BB[1][3],angle)

# 切割倾斜矩形
def cut_pic(pic_name , pic_labels , pic_wordBB , pic_charBB, prefix_store_path, img_root):
    if len(pic_labels) == 1:
        return [],[]
    pic_path_list = []
    pic_label_list = []
    rec_pic_name = ''
    for b in pic_name.tostring().decode('utf-8'):
        if b != b'\x00'.decode('utf-8'):
            rec_pic_name = rec_pic_name + b

    img_path = img_root + rec_pic_name
    img = Image.open(img_path)
    #img.show()
    i = 0
    for i_label in pic_labels:

        width = img.size[0]
        height = img.size[1]

        rotate_angle = get_angle(int(pic_wordBB[i_label][0][0]),int(pic_wordBB[i_label][1][0]),int(pic_wordBB[i_label][0][2]),int(pic_wordBB[i_label][1][2]),int(pic_wordBB[i_label][0][3]),int(pic_wordBB[i_label][1][3]))
        rotated_img = img.rotate(rotate_angle)
        get_all_rotated_point(width/2,height/2,pic_wordBB[i_label],rotate_angle)
        
        left = min(int(pic_wordBB[i_label][0][0]),int(pic_wordBB[i_label][0][1]),int(pic_wordBB[i_label][0][2]),int(pic_wordBB[i_label][0][3]))-5
        right = max(int(pic_wordBB[i_label][0][0]),int(pic_wordBB[i_label][0][1]),int(pic_wordBB[i_label][0][2]),int(pic_wordBB[i_label][0][3]))+5
        up = min(int(pic_wordBB[i_label][1][0]),int(pic_wordBB[i_label][1][1]),int(pic_wordBB[i_label][1][2]),int(pic_wordBB[i_label][1][3]))-5
        down = max(int(pic_wordBB[i_label][1][0]),int(pic_wordBB[i_label][1][1]),int(pic_wordBB[i_label][1][2]),int(pic_wordBB[i_label][1][3]))+5
        cropped = rotated_img.crop((left,up,right,down))

        i_char = 0
        for _ in i_label:
            get_all_rotated_point(width/2,height/2,pic_charBB[i_label,i_char],rotate_angle)
            get_all_cropped_point(up,left,pic_charBB[i_label,i_char])
            i_char += 1 

        if not cropped_point_verify(pic_charBB[i_label,0]):
            continue           

        store_path = rec_pic_name.split('.')[0].split('/')[1] + '_' + str(i)+ '_' + i_label + '.png'
        store_path = prefix_store_path + '/' + store_path
        pic_path_list.append(store_path)
        pic_label_list.append(i_label)
        cropped.save(store_path)

        # debug_charBB_rotated(store_path,pic_charBB,i_label)
        
        i += 1
    return pic_path_list, pic_label_list

# 用来测试字符级BB在图像旋转后是否准确
# 简陋的写法，基本只能用在上面,根据需要使用下面的两种
# cropped.save(store_path),rotated_img.save(store_path)
def debug_charBB_rotated(store_path,pic_charBB,i_label):
        im = array(Image.open(store_path))
        imshow(im)
        i_char = 0
        for _ in i_label:
            x = [pic_charBB[i_label,i_char][0][0],pic_charBB[i_label,i_char][0][1],pic_charBB[i_label,i_char][0][2],pic_charBB[i_label,i_char][0][3],pic_charBB[i_label,i_char][0][0]]
            y = [pic_charBB[i_label,i_char][1][0],pic_charBB[i_label,i_char][1][1],pic_charBB[i_label,i_char][1][2],pic_charBB[i_label,i_char][1][3],pic_charBB[i_label,i_char][1][0]]
            i_char += 1
            plot(x,y,'b')
        title('i')
        pylab.show()

def get_all_cropped_point(up,left,BB):
    BB[0][0], BB[1][0] = get_croped_point(up,left,BB[0][0],BB[1][0])
    BB[0][1], BB[1][1] = get_croped_point(up,left,BB[0][1],BB[1][1])
    BB[0][2], BB[1][2] = get_croped_point(up,left,BB[0][2],BB[1][2])
    BB[0][3], BB[1][3] = get_croped_point(up,left,BB[0][3],BB[1][3])

def get_croped_point(up,left,x,y):
    fx = x-left
    fy = y-up
    return fx,fy

def cropped_point_verify(BB):
    if BB[1][0] < 0:
        return False
    else:
        return True

# 查看输出的格式
def see_dtype(pic_name , pic_labels , pic_wordBB , pic_charBB):
    print('dtype pic name:',pic_name.dtype) #  <U18 
    print('dtype pic name:',pic_name.dtype.name) # str576
    print('pic name:',pic_name.tostring()) #bytes
    print('pic name:',pic_name.tostring().decode('utf-8')) # string
    # 我佛了..最开始是numpy的ndarray类型的，所以要先用tostring转换为bytes，再decode才能得到最终的string
    print('dtype pic label:',pic_labels[0].dtype)  # str
    print('dtype pic BB:',pic_wordBB[pic_labels[0]][0][0].dtype) #  float32

# 切割好的图片放入data，并以create_lmdb_dataset要求的格式存储
def build_lmdb_label_mask(pic_name , pic_labels , pic_wordBB , pic_charBB, prefix_store_path, img_root):
    path_list, label_list, mask_path_last, bb_list = cut_pic_old(pic_name , pic_labels , pic_wordBB , pic_charBB, prefix_store_path, img_root)
    for path,label,mask_path,bb in zip(path_list,label_list,mask_path_last,bb_list):
        txt_path = prefix_store_path + '/' + 'gt.txt'
        with open (txt_path,'a') as f:
            f.write(path + '\t' + label + '\t' + mask_path + '\t' + bb + '\n')

def get_gt_mask(img,pic_wordBB,pic_charBB,i_label,store_mask_path):
    mask = np.zeros(img.shape)
    i_char = 0
    for _ in i_label:
        b = np.array([[[pic_charBB[i_label,i_char][0][0],pic_charBB[i_label,i_char][1][0]],
            [pic_charBB[i_label,i_char][0][1],pic_charBB[i_label,i_char][1][1]],
            [pic_charBB[i_label,i_char][0][2],pic_charBB[i_label,i_char][1][2]],
            [pic_charBB[i_label,i_char][0][3],pic_charBB[i_label,i_char][1][3]]]],dtype = np.int32)
        dimx,dimy = get_direction(pic_charBB[i_label,i_char][0][1],pic_charBB[i_label,i_char][1][1],pic_charBB[i_label,i_char][0][2],pic_charBB[i_label,i_char][1][2])
        cv2.polylines(mask,b,1,(255,dimx,dimy))
        cv2.fillPoly(mask,b,(255,dimx,dimy))
        i_char += 1
    cv2.imwrite(store_mask_path, mask)

def get_gt_mask_1dim(img,pic_wordBB,pic_charBB,i_label,store_mask_path):
    mask = np.zeros(img.shape)
    i_char = 0
    for _ in i_label:
        b = np.array([[[pic_charBB[i_label,i_char][0][0],pic_charBB[i_label,i_char][1][0]],
            [pic_charBB[i_label,i_char][0][1],pic_charBB[i_label,i_char][1][1]],
            [pic_charBB[i_label,i_char][0][2],pic_charBB[i_label,i_char][1][2]],
            [pic_charBB[i_label,i_char][0][3],pic_charBB[i_label,i_char][1][3]]]],dtype = np.int32)
        dimx,dimy = get_direction(pic_charBB[i_label,i_char][0][1],pic_charBB[i_label,i_char][1][1],pic_charBB[i_label,i_char][0][2],pic_charBB[i_label,i_char][1][2])
        cv2.polylines(mask,b,1,(255,dimx,dimy))
        cv2.fillPoly(mask,b,(255,dimx,dimy))
        i_char += 1
    cv2.imwrite(store_mask_path, mask)
    
def get_direction(x2,y2,x3,y3):
    z = math.sqrt((x2-x3)**2+(y2-y3)**2)
    alpha = math.atan((y2-y3)/(x2-x3))*180/math.pi
    cos = (x2-x3)/z
    sin = (y3-y2)/z
    #print(alpha)
    #print(cos,sin)
    return (100+cos*100),(100+sin*100)