from unittest import result
import cv2
import numpy as np
import os
import glob

import matplotlib.pyplot as plt

img1_path = './result_frames1/ReReVST-candy-ambush_5/*.png'

result_paths = './heat_map/'
if not os.path.exists(result_paths):
    os.mkdir(result_paths)

    
frame_list = glob.glob(img1_path)
frame_num = len(frame_list)

for i in range(frame_num):



    img1 = cv2.imread(frame_list[i])
    n = int(frame_list[i].split('/frame_00')[-1].split('.')[0])+1
    if(n<=50):
        n = str("%04d"% n)

        img2_path = './result_frames1/ReReVST-candy-ambush_5/frame_'+n+'.png'
        img2 = cv2.imread(img2_path)
        # img3 = cv2.imread('./res2.png')
        print(i)
        print(frame_list[i])
        print(img1_path)
        print(img2_path)


        res = cv2.absdiff(img1, img2)
        # res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        # res1 = cv2.add(img3, res)
        # cv2.imwrite('./res.png', res)

        res = cv2.applyColorMap(res, 11)

        cv2.imwrite('./heat_map/res'+n+'.png', res) 
    
    
