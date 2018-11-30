import pydicom as dicom
import os
import cv2
import sys
import  numpy as np
import  matplotlib.pyplot as plt
DIR='data/maskroi/'
originpath='000000.dcm'
maskpath  ='000001.dcm'

def dcm_as_numpy_array(filepath):
    if not os.path.isfile(filepath):
        print "no picture"
        return None,None
    ds=dicom.dcmread(filepath)
    imgarray=ds.pixel_array
    shape=imgarray.shape
    return imgarray,shape

maskname=DIR+maskpath
originname=DIR+originpath

def get_mask_roi(filepath):
    img_mask, shape = dcm_as_numpy_array(filepath)
    boundsize = 50
    _, cnts, _ = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # cv2.drawContours(img_origin,c,-1,(0,255,0),3)
    x, y, w, h = cv2.boundingRect(c)
    x=x-boundsize
    y=y-boundsize
    w=w+2*boundsize
    h=h+2*boundsize
    img_mask = cv2.resize(img_mask, (1000, 1000))
    cv2.imwrite("mask.png",img_mask)
    return filepath,x,y,w,h



img_origin,shape=dcm_as_numpy_array(originname)
img_mask,x,y,w,h=get_mask_roi(maskname)
print x,y,w,h
img=cv2.rectangle(img_origin,(x,y),(x+w,y+h),(0,255,0),2)
img_origin=cv2.resize(img_origin,(1000,1000))
cv2.imshow("orgin1",img_origin)
cv2.imwrite("origin.png",img_origin)
cv2.waitKey()




