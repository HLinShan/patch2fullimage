# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:20:34 2018

@author: wyw
"""

import os
from os.path import join, isdir
import pydicom as dicom
import cv2
import csv

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

def get_the_directory_under(dirpath):
    """ 
    Get the first directory under a path.
    """
    dirs = [name for name in os.listdir(dirpath) if isdir(join(dirpath, name))]
    return dirs[0]

def dig_out_dcm_image(dirpath, nb_layers=2):
    """ 
    Remove the useless directory layers, get a filepath to the
    dicom image file.
    """

    # There should be 3 layers of useless dirs
    onion = dirpath

    for _ in range(nb_layers):
        onion_peel = get_the_directory_under(onion)
        onion = join(onion, onion_peel)
    return onion

def get_mask_roi(img_mask):
    '''
    功能：获得mask中分割出的roi的矩形左上坐标(x,y)和矩形长宽(w,h)
    img_mask: mask array
    '''
    boundsize = 25
    _, cnts, _ = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # cv2.drawContours(img_origin,c,-1,(0,255,0),3)
    x, y, w, h = cv2.boundingRect(c)
    x=x-boundsize
    y=y-boundsize
    w=w+2*boundsize
    h=h+2*boundsize
    return x,y,w,h

#数据存放文件夹目录
data_dir = "..\data\CBIS-DDSM"
#写入csv文件的抬头第一行内容
headers = ['image_filepath', 'mask_filepath', 'mask_filename', 'pathology', 'itype', 'x', 'y', 'w', 'h']
#存放CBIS-DDSM提供的原始csv文件的文件夹目录
csv_dir = '..\labels_raw_csv'
#DICOM文件名
dicom_imgname = [
        "000000.dcm",
        "000001.dcm"
        ]
#CBIS-DDSM提供的原始csv文件名
csv_file = [
#        'Calc-Test.csv',
#        'Calc-Training.csv',
        'Mass-Test.csv',
        'Mass-Training.csv'
        ]

#写入roi.csv文件
with open('roi.csv', 'a+', newline='') as f:
    #创建csv字典写入器，headers是抬头
    writer = csv.DictWriter(f, headers)
    #第一行写抬头
#    writer.writeheader()
    #读取原始四份csv文件
    for cfile in csv_file:
        cfile = join(csv_dir, cfile)
        
        with open(cfile, newline='') as csvfile:
            #创建csv字典读取器            
            csv_reader = csv.DictReader(csvfile)
            #逐行读入
            rows = [row for row in csv_reader]
            #取出不正常类型
            cropped_roi_itype = [row['abnormality type'] for row in rows[1:]]
            #取出病理结果
            cropped_roi_pathology = [row['pathology'] for row in rows[1:]]
            #取出原图路径
            full_image_full_path = [dig_out_dcm_image(join(data_dir, row['image file path'].split('/')[0])) for row in rows[1:]]
            #取出mask路径
            mask_full_path = [dig_out_dcm_image(join(data_dir, row['ROI mask file path'].split('/')[0])) for row in rows[1:]]
            #遍历所有mask
            for (fullpath, maskpath, pathology, itype) in zip(full_image_full_path, mask_full_path, cropped_roi_pathology, cropped_roi_itype):
                #全图路径
                imgpath_dcm = join(fullpath, dicom_imgname[0])
                #读入全图
                full_imgarray, shape, full_filename = dcm_as_numpy_array(imgpath_dcm)
                #mask路径
                filepath_dcm = join(maskpath, dicom_imgname[1])
                #读入mask（000001.dcm)
                mask_imgarray, shape, mask_filename = dcm_as_numpy_array(filepath_dcm)
                #判断是否成功读入文件，若失败，则continue
                if shape == None:
                    continue
                #判断读入文件大小是否小于4200，若是，则需重新读000000.dcm
                if shape[0] < 4200:
                    filepath_dcm = join(maskpath, dicom_imgname[0])
                    mask_imgarray, shape, mask_filename = dcm_as_numpy_array(filepath_dcm)
                #再次判断是否成功读入文件以及文件大小是否小于4200
                if shape == None or shape[0] < 4200:
                    continue
                #寻找x，y，w，h
                x,y,w,h = get_mask_roi(mask_imgarray)
                
                print(maskpath)
                #写入csv文件
                writer.writerow({'image_filepath': fullpath, 
                                 'mask_filepath': maskpath, 
                                 'mask_filename': mask_filename, 
                                 'pathology': pathology, 
                                 'itype': itype, 
                                 'x': x, 
                                 'y': y, 
                                 'w': w, 
                                 'h': h})
    
    print('Write roi.csv done!')    