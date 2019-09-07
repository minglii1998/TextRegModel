from get_mat_data import get_pic_info
from get_mat_data import build_lmdb_label_mask
import scipy.io as scio
import os

import numpy as np

mat_path = "/home/liming/data/SynthText/gt.mat"
img_root = "/home/liming/data/SynthText/"
store_prepared_path = "/home/liming/data/ST_lmdb_mask_bb_prep/"

if os.path.exists(store_prepared_path):
    pass
else:
    os.mkdir(store_prepared_path)

data = scio.loadmat(mat_path)


for i in range(858750):
    print('pic:',i)
    pic_name , pic_labels , pic_wordBB , pic_charBB = get_pic_info(data,i)
    try:
        build_lmdb_label_mask(pic_name , pic_labels , pic_wordBB , pic_charBB, store_prepared_path, img_root)
    except ValueError: 
        continue
