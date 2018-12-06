# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:31:14 2018

@author: hls
"""


import os.path as osp
import csv
import os
from os.path import join, isdir

data_dir='calc-training-out-png'
TXT_FILE='Calc-Training-fullimg.txt'
CSV_FILE='Calc-Training.csv'
patho = [
    'BENIGN',
    'MALIGNANT',
    'BENIGN_WITHOUT_CALLBACK']
ity = [
    'calcification',
    'mass']



# dirpath = '.\data\patch-test-wyw'
# number of each class
num_classes = [
    0, 0
]
with open(TXT_FILE, 'w') as ltxt:
    with open(CSV_FILE) as csvfile:

        # 创建csv字典读取器
        csv_reader = csv.DictReader(csvfile)
        # 逐行读入
        rows = [row for row in csv_reader]
        pathologys = [row['pathology'] for row in rows[1:]]
        # 取出原图路径
        full_image_full_path = [join(data_dir, row['image file path'].split('/')[0]) for row in
                                rows[1:]]



        for (pathology,fullname) in zip(pathologys, full_image_full_path):
            print pathology
            print fullname

            if pathology == patho[0]:
                ltxt.write(full_image_full_path[0] + '.png 0\n')

                # num_classes[0] += 1
            if pathology == patho[1]:
                ltxt.write(full_image_full_path[0] + '.png 1\n')

                # num_classes[1] += 1
            # else:print "not"

        # for num in num_classes:
        #     print(num)
