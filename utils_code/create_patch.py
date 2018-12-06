# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 20:41:24 2018

@author: wyw
create patch :uising roi (x,y,w,h) from maskimg in fullimage  get patch

"""

import os
from os.path import join
import csv
import pydicom as dicom
import cv2
import numpy as np

def dcm_as_numpy_array(filepath):
    '''
    读入dicom文件，返回array，shape和文件名
    filepath: dicom文件路径
    '''
    if not os.path.isfile(filepath):
        print ("no picture")
        return None,None,None
    ds=dicom.dcmread(filepath)
    imgarray=ds.pixel_array
    shape=imgarray.shape
    return imgarray, shape, filepath.split('\\')[-1]

#csv文件的抬头内容
headers = ['image_filepath', 'mask_filepath', 'mask_filename', 'pathology', 'itype', 'x', 'y', 'w', 'h']
#DICOM文件名
dicom_imgname = [
        "000000.dcm",
        "000001.dcm"
        ]
#保存图片文件夹路径
save_dir = '.\data\\'
#可视化标志位
visualization = False
#断点
start = 600
end = -1
#读取roi.csv文件
with open('roi2.csv', newline = '') as csvfile:
    #创建csv字典读取器            
    csv_reader = csv.DictReader(csvfile)
    #逐行读入
    i = start
    rows = [row for row in csv_reader]
    image_filepaths = [row[headers[0]] for row in rows[start:end]]
    mask_filepaths = [row[headers[1]] for row in rows[start:end]]
    mask_filenames = [row[headers[2]] for row in rows[start:end]]
    pathologys = [row[headers[3]] for row in rows[start:end]]
    itypes = [row[headers[4]] for row in rows[start:end]]
    xs = [int(row[headers[5]]) for row in rows[start:end]]
    ys = [int(row[headers[6]]) for row in rows[start:end]]
    ws = [int(row[headers[7]]) for row in rows[start:end]]
    hs = [int(row[headers[8]]) for row in rows[start:end]]
    
    for image_filepath, mask_filepath, mask_filename, pathology, itype, x, y, w, h in zip(
            image_filepaths, mask_filepaths, mask_filenames, pathologys, itypes, xs, ys, ws, hs):
        
        #全图路径
        imgpath_dcm = join(image_filepath, dicom_imgname[0])
        #读入全图
        full_imgarray, shape, full_filename = dcm_as_numpy_array(imgpath_dcm)
        #判断坐标是否超出图像大小
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        x2 = x + w
        if x2 > shape[0]:
            x2 = shape[0]
        y2 = y + h        
        if x2 > shape[1]:
            x2 = shape[1]
        #提取roi区域
        roi_imgarray = full_imgarray[y:y2, x:x2]
        #可视化输出全图、mask、roi
        if visualization:
            #mask路径
            maskpath_dcm = join(mask_filepath, mask_filename)
            #读入mask
            mask_imgarray, shape, mask_filename = dcm_as_numpy_array(maskpath_dcm)
            mask_imgarray=cv2.resize(mask_imgarray,(1000,1000))
            full_imgarray=cv2.rectangle(full_imgarray,(x,y),(x+w,y+h),(0,255,0),2)
            full_imgarray=cv2.resize(full_imgarray,(1000,1000))
            cv2.imshow("orgin",full_imgarray)
            cv2.imshow("mask",mask_imgarray)
            cv2.imshow("roi",roi_imgarray)
            cv2.waitKey()
        #保存文件名
        save_name = 'patch-'+mask_filepath.split('\\')[3].split('-')[-1].split('_')[0].lower()+'-wyw'
        save_name = join(save_dir, save_name, mask_filepath.split('\\')[3]) + '.png'
        i += 1
        print(save_name, i)
        #写PNG图片，第三个参数是保存为png格式，压缩率为0，也就是无损失保存
        cv2.imwrite(save_name, roi_imgarray, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) 
        