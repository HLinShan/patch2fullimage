# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:31:14 2018

@author: wyw
"""

import os
import os.path as osp
import csv

patho = [
        'BENIGN',
        'MALIGNANT',
        'BENIGN_WITHOUT_CALLBACK']
ity = [
       'calcification',
       'mass']
#写入csv文件的抬头第一行内容
headers = ['image_filepath', 'mask_filepath', 'mask_filename', 'pathology', 'itype', 'x', 'y', 'w', 'h']
dirpath = '.\data\patch-training-wyw'
#dirpath = '.\data\patch-test-wyw'
#number of each class
num_classes = [
        0,0,0,0
        ]
with open('train_patch_wyw.txt', 'w') as ltxt:
    
    with open('roi2.csv', newline = '') as csvfile:
    
        #创建csv字典读取器            
        csv_reader = csv.DictReader(csvfile)            
        #逐行读入
        rows = [row for row in csv_reader]
        mask_names = [row[headers[1]].split('\\')[3] for row in rows[1:]]
        pathologys = [row[headers[3]] for row in rows[1:]]
        itypes = [row[headers[4]] for row in rows[1:]]
        for name in os.listdir(dirpath):            
            
            name = name.split('.')[0]
            for(maskname, pathology, itype) in zip(mask_names, pathologys, itypes):
                if maskname == name:
                    if pathology == patho[0]:
                        if itype == ity[0]:
                            ltxt.write(name + '.png 0\n')
                            num_classes[0] += 1
                        else:                            
                            ltxt.write(name + '.png 1\n')
                            num_classes[1] += 1
                    elif pathology == patho[1]:
                        if itype == ity[0]:
                            ltxt.write(name + '.png 2\n')
                            num_classes[2] += 1
                        else:                            
                            ltxt.write(name + '.png 3\n')
                            num_classes[3] += 1
                    else:
                        break
                    break
        for num in num_classes:
            print(num)
        
        