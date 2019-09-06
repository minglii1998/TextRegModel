# pth采纳数太大了，高达281G。然后又用不到，就删掉好了

import os

delete_path = "/home/liming/code/expr_rightloss_2gpu/"
delete_file_eg = "netRES_4_102000.pth"

for i in range (500,106501,500):
    delete_file = delete_path + "netRES_4_" + str(i) + ".pth"
    if os.path.exists(delete_file):
        os.remove(delete_file)
        print('delete:',delete_file)
