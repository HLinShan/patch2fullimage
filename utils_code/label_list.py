# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:31:14 2018

@author: wyw
"""

import os
import os.path as osp

label = [
        'background',
        'calc_ben',
        'calc_mal',
        'mass_ben',
        'mass_mal']

dirpath = 'patch-train-pathology'
with open('train_pathology_list.txt', 'w') as ltxt:
    
    for p, lab in enumerate(label):
        
        path = osp.join(dirpath, lab)
        for name in os.listdir(path):
            filepath = osp.join(path, name)
            ltxt.write(filepath + ' ' + str(p) + '\n')
        
        