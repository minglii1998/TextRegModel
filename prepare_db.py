from get_mat_data import get_pic_info
from get_mat_data import build_lmdb_only_label
import scipy.io as scio
import os

mat_path = "/home/liming/data/SynthText/gt.mat"
img_root = "/home/liming/data/SynthText/"
store_prepared_path = "/home/liming/data/ST_4lmdb"

os.mkdir(store_prepared_path)
data = scio.loadmat(mat_path)

for i in range(5):
    pic_name , pic_labels , pic_wordBB , pic_charBB = get_pic_info(data,i)
    build_lmdb_only_label(pic_name , pic_labels , pic_wordBB , pic_charBB, store_prepared_path, img_root)
