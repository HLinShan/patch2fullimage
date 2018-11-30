# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:40:57 2018

@author: wyw
"""

import os, sys
from os import path
from os.path import join, isdir, isfile
import numpy as np
import csv
# from util.path import get_the_directory_under
import pydicom as dicom

from matplotlib import pyplot as plt
from scipy.misc import imsave, imresize
# from util.image import normalize_between
from scipy.misc import toimage
import cv2

data_dir = '..\data\CBIS-DDSM'
csv_dir = '..\labels_raw_csv'
dicom_imgname = [
        "000000.dcm",
        "000001.dcm"
        ]
csv_file = [
#        'Calc-Test.csv',
#        'Calc-Training.csv',
        'Mass-Test.csv',
#        'Mass-Training.csv'
        ]

def crop_img(img, bbox):
    '''Crop an image using bounding box
    '''
    x,y,w,h = bbox
    return img[y:y+h, x:x+w]

def add_img_margins(img, margin_size):
    '''Add all zero margins to an image
    '''
    enlarged_img = np.zeros((img.shape[0]+int(margin_size*2), 
                             img.shape[1]+int(margin_size*2)))
    enlarged_img[margin_size:margin_size+img.shape[0], 
                 margin_size:margin_size+img.shape[1]] = img
    return enlarged_img

def read_resize_img(fname, target_size=None, target_height=None, 
                    target_scale=None, gs_255=False, rescale_factor=None):
    '''Read an image (.png, .jpg, .dcm) and resize it to target size.
    '''
    if target_size is None and target_height is None:
        raise Exception('One of [target_size, target_height] must not be None')
    if path.splitext(fname)[1] == '.dcm':
        img = dicom.read_file(fname).pixel_array
    else:
        if gs_255:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if target_height is not None:
        target_width = int(float(target_height)/img.shape[0]*img.shape[1])
    else:
        target_height, target_width = target_size
    if (target_height, target_width) != img.shape:
        img = cv2.resize(
            img, dsize=(target_width, target_height), 
            interpolation=cv2.INTER_CUBIC)
    img = img.astype('float32')
    if target_scale is not None:
        img_max = img.max() if img.max() != 0 else target_scale
        img *= target_scale/img_max
    if rescale_factor is not None:
        img *= rescale_factor
    return img

#### Define some functions to use ####
def const_filename(pat, side, view, directory, itype=None, abn=None):
    token_list = [pat, side, view]
    if itype is not None:
        token_list.insert(
            0, ('Calc' if itype == 'calcification' else 'Mass') + '-Training')
        token_list.append(str(abn))
    fn = "_".join(token_list) + ".png"
    return os.path.join(directory, fn)

def crop_val(v, minv, maxv):
    v = v if v >= minv else minv
    v = v if v <= maxv else maxv
    return v

def overlap_patch_roi(patch_center, patch_size, roi_mask, 
                      add_val=1000, cutoff=.5):
    x1,y1 = (patch_center[0] - patch_size//2, 
             patch_center[1] - patch_size//2)
    x2,y2 = (patch_center[0] + patch_size//2, 
             patch_center[1] + patch_size//2)
    x1 = crop_val(x1, 0, roi_mask.shape[1])
    y1 = crop_val(y1, 0, roi_mask.shape[0])
    x2 = crop_val(x2, 0, roi_mask.shape[1])
    y2 = crop_val(y2, 0, roi_mask.shape[0])
    roi_area = (roi_mask>0).sum()
    roi_patch_added = roi_mask.copy()
    roi_patch_added[y1:y2, x1:x2] += add_val
    patch_area = (roi_patch_added>=add_val).sum()
    inter_area = (roi_patch_added>add_val).sum().astype('float32')
    return (inter_area/roi_area > cutoff or inter_area/patch_area > cutoff)

def create_blob_detector(roi_size=(128, 128), blob_min_area=3, 
                         blob_min_int=.5, blob_max_int=.95, blob_th_step=10):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = blob_min_area
    params.maxArea = roi_size[0]*roi_size[1]
    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False
    # blob detection only works with "uint8" images.
    params.minThreshold = int(blob_min_int*255)
    params.maxThreshold = int(blob_max_int*255)
    params.thresholdStep = blob_th_step
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params)


def sample_patches(img, roi_mask, out_dir, img_id, abn, pos, patch_size=256,
                   pos_cutoff=.75, neg_cutoff=.35,
                   nb_bkg=100, nb_abn=100, start_sample_nb=0, itype='calcification',
                   bkg_dir='background', 
                   calc_pos_dir='calc_mal', calc_neg_dir='calc_ben', 
                   mass_pos_dir='mass_mal', mass_neg_dir='mass_ben', 
                   verbose=False):
    if pos:
        if itype == 'calcification':
            roi_out = os.path.join(out_dir, calc_pos_dir)
        else:
            roi_out = os.path.join(out_dir, mass_pos_dir)
    else:
        if itype == 'calcification':
            roi_out = os.path.join(out_dir, calc_neg_dir)
        else:
            roi_out = os.path.join(out_dir, mass_neg_dir)
    bkg_out = os.path.join(out_dir, bkg_dir)
    basename = '_'.join([img_id, str(abn)])

    img = add_img_margins(img, patch_size//2)
    roi_mask = add_img_margins(roi_mask, patch_size//2)
    # Get ROI bounding box.
    roi_mask_8u = roi_mask.astype('uint8')
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        contours,_ = cv2.findContours(
            roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _,contours,_ = cv2.findContours(
            roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)  # find the largest contour.
    rx,ry,rw,rh = cv2.boundingRect(contours[idx])
    if verbose:
        M = cv2.moments(contours[idx])
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print ("ROI centroid=", (cx,cy)); sys.stdout.flush()
        except ZeroDivisionError:
            cx = rx + int(rw/2)
            cy = ry + int(rh/2)
            print ("ROI centroid=Unknown, use b-box center=", (cx,cy))
            sys.stdout.flush()

    rng = np.random.RandomState(12345)
    # Sample abnormality first.
    sampled_abn = 0
    nb_try = 0
    while sampled_abn < nb_abn:
        if nb_abn > 1:
            x = rng.randint(rx, rx + rw)
            y = rng.randint(ry, ry + rh)
            nb_try += 1
            if nb_try >= 1000:
                break
                print ("Nb of trials reached maximum, decrease overlap cutoff by 0.05")
                sys.stdout.flush()
                pos_cutoff -= .05
                nb_try = 0
                if pos_cutoff <= .0:
                    raise Exception("overlap cutoff becomes non-positive, "
                                    "check roi mask input.")
        else:
            x = cx
            y = cy
        # import pdb; pdb.set_trace()
        if nb_abn == 1 or overlap_patch_roi((x,y), patch_size, roi_mask, 
                                            cutoff=pos_cutoff):
            patch = img[y - patch_size//2:y + patch_size//2, 
                        x - patch_size//2:x + patch_size//2]
            patch = patch.astype('int32')
            patch_img = toimage(patch, high=patch.max(), low=patch.min(), 
                                mode='I')
            # patch = patch.reshape((patch.shape[0], patch.shape[1], 1))
            filename = basename + "_%04d" % (sampled_abn) + ".png"
            fullname = os.path.join(roi_out, filename)
            # import pdb; pdb.set_trace()
            patch_img.save(fullname)
            sampled_abn += 1
            print(sampled_abn)
            nb_try = 0
            if verbose:
                print ("sampled an abn patch at (x,y) center=", (x,y))
                sys.stdout.flush()
    # Sample background.
    sampled_bkg = start_sample_nb
    while sampled_bkg < start_sample_nb + nb_bkg:
        x = rng.randint(patch_size//2, img.shape[1] - patch_size//2)
        y = rng.randint(patch_size//2, img.shape[0] - patch_size//2)
        if not overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=neg_cutoff):
            patch = img[y - patch_size//2:y + patch_size//2, 
                        x - patch_size//2:x + patch_size//2]
            patch = patch.astype('int32')
            patch_img = toimage(patch, high=patch.max(), low=patch.min(), 
                                mode='I')
            filename = basename + "_%04d" % (sampled_bkg) + ".png"
            fullname = os.path.join(bkg_out, filename)
            patch_img.save(fullname)
            sampled_bkg += 1
            if verbose:
                print ("sampled a bkg patch at (x,y) center=", (x,y))
                sys.stdout.flush()

def sample_hard_negatives(img, roi_mask, out_dir, img_id, abn,  
                          patch_size=256, neg_cutoff=.35, nb_bkg=100, 
                          start_sample_nb=0,
                          bkg_dir='background', verbose=False):
    '''WARNING: the definition of hns may be problematic.
    There has been study showing that the context of an ROI is also useful
    for classification.
    '''
    bkg_out = os.path.join(out_dir, bkg_dir)
    basename = '_'.join([img_id, str(abn)])

    img = add_img_margins(img, patch_size//2)
    roi_mask = add_img_margins(roi_mask, patch_size//2)
    # Get ROI bounding box.
    roi_mask_8u = roi_mask.astype('uint8')
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        contours,_ = cv2.findContours(
            roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _,contours,_ = cv2.findContours(
            roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)  # find the largest contour.
    rx,ry,rw,rh = cv2.boundingRect(contours[idx])
    if verbose:
        M = cv2.moments(contours[idx])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print ("ROI centroid=", (cx,cy)); sys.stdout.flush()

    rng = np.random.RandomState(12345)
    # Sample hard negative samples.
    sampled_bkg = start_sample_nb
    while sampled_bkg < start_sample_nb + nb_bkg:
        x1,x2 = (rx - patch_size/2, rx + rw + patch_size/2)
        y1,y2 = (ry - patch_size/2, ry + rh + patch_size/2)
        x1 = crop_val(x1, patch_size/2, img.shape[1] - patch_size/2)
        x2 = crop_val(x2, patch_size/2, img.shape[1] - patch_size/2)
        y1 = crop_val(y1, patch_size/2, img.shape[0] - patch_size/2)
        y2 = crop_val(y2, patch_size/2, img.shape[0] - patch_size/2)
        x = rng.randint(x1, x2)
        y = rng.randint(y1, y2)
        if not overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=neg_cutoff):
            patch = img[y - patch_size/2:y + patch_size/2, 
                        x - patch_size/2:x + patch_size/2]
            patch = patch.astype('int32')
            patch_img = toimage(patch, high=patch.max(), low=patch.min(), 
                                mode='I')
            filename = basename + "_%04d" % (sampled_bkg) + ".png"
            fullname = os.path.join(bkg_out, filename)
            patch_img.save(fullname)
            sampled_bkg += 1
            if verbose:
                print ("sampled a hns patch at (x,y) center=", (x,y))
                sys.stdout.flush()

def sample_blob_negatives(img, roi_mask, out_dir, img_id, abn, blob_detector, 
                          patch_size=256, neg_cutoff=.35, nb_bkg=100, 
                          start_sample_nb=0,
                          bkg_dir='background', verbose=False):
    bkg_out = os.path.join(out_dir, bkg_dir)
    basename = '_'.join([img_id, str(abn)])

    img = add_img_margins(img, patch_size/2)
    roi_mask = add_img_margins(roi_mask, patch_size/2)
    # Get ROI bounding box.
    roi_mask_8u = roi_mask.astype('uint8')
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        contours,_ = cv2.findContours(
            roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _,contours,_ = cv2.findContours(
            roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)  # find the largest contour.
    rx,ry,rw,rh = cv2.boundingRect(contours[idx])
    if verbose:
        M = cv2.moments(contours[idx])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print ("ROI centroid=", (cx,cy)); sys.stdout.flush()

    # Sample blob negative samples.
    key_pts = blob_detector.detect((img/img.max()*255).astype('uint8'))
    rng = np.random.RandomState(12345)
    key_pts = rng.permutation(key_pts)
    sampled_bkg = 0
    for kp in key_pts:
        if sampled_bkg >= nb_bkg:
            break
        x,y = int(kp.pt[0]), int(kp.pt[1])
        if not overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=neg_cutoff):
            patch = img[y - patch_size/2:y + patch_size/2, 
                        x - patch_size/2:x + patch_size/2]
            patch = patch.astype('int32')
            patch_img = toimage(patch, high=patch.max(), low=patch.min(), 
                                mode='I')
            filename = basename + "_%04d" % (start_sample_nb + sampled_bkg) + ".png"
            fullname = os.path.join(bkg_out, filename)
            patch_img.save(fullname)
            if verbose:
                print ("sampled a blob patch at (x,y) center=", (x,y))
                sys.stdout.flush()
            sampled_bkg += 1
    return sampled_bkg

#### End of function definition ####
    
def dig_out_dcm_image(dirpath, nb_layers=2):
    """ Remove the useless directory layers, get a filepath to the
        dicom image file.
    """

    # There should be 3 layers of useless dirs
    #metadata = []
    onion = dirpath

    for _ in range(nb_layers):
        onion_peel = get_the_directory_under(onion)
        onion = join(onion, onion_peel)
        #metadata.append(onion_peel)
#    filepath = join(onion, dicom_imgname)
    return onion

def dcm_as_numpy_array(filepath):
    if not isfile(filepath):
        print("Not a dir, continuing")
        return None, None
    ds = dicom.dcmread(filepath)
    imgarray = ds.pixel_array
    #print("imgarray:\n", imgarray, type(imgarray), imgarray.shape)
    shape = imgarray.shape
    return imgarray, shape

def save_plot_of_datapoint_shapes(shape_datapts, plot_save_path):
    plt.plot(*zip(*shape_datapts), 'bo')
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.savefig(plot_save_path)

def preprocess(imgarray, resize = None):
    if resize is not None:
        size = (resize, resize)
        imgarray = imresize(imgarray, size, interp='bicubic', mode='F')
    imgarray = normalize_between(imgarray, 0, 1)
    imgarray = imgarray.astype(np.float32)
    
    return imgarray
    
    
        
if __name__ == "__main__":
    
    idxstart = 33
    idxend = 379
    for cfile in csv_file:
        cfile = join(csv_dir, cfile)
        
        with open(cfile, newline='') as csvfile:
                        
            name = cfile.split('\\')[-1].split('.')[0]
            
            save_dir = '.\\data\\patch-test-pathology'            
            i = idxstart
            
            csv_reader = csv.DictReader(csvfile)
            rows = [row for row in csv_reader]
    #            cropped_roi_path = [row['ROI mask file path'] for row in rows[1:]]
            file_name = [row['ROI mask file path'].split('/')[0] for row in rows[idxstart:idxend]]
            cropped_roi_abn = [row['abnormality id'] for row in rows[idxstart:idxend]]
            cropped_roi_itype = [row['abnormality type'] for row in rows[idxstart:idxend]]
            cropped_roi_pos = [row['pathology'] for row in rows[idxstart:idxend]]
            full_image_full_path = [dig_out_dcm_image(join(data_dir, row['image file path'].split('/')[0])) for row in rows[idxstart:idxend]]
            cropped_roi_full_path = [dig_out_dcm_image(join(data_dir, row['ROI mask file path'].split('/')[0])) for row in rows[idxstart:idxend]]
            for (fullpath, croppath, filename, abn, posf, itype) in zip(full_image_full_path, cropped_roi_full_path, file_name, cropped_roi_abn, cropped_roi_pos, cropped_roi_itype):
                print(filename, i, posf)
                i += 1
                imgpath_dcm = join(fullpath, dicom_imgname[0])
                full_imgarray, shape = dcm_as_numpy_array(imgpath_dcm)
                filepath_dcm = join(croppath, dicom_imgname[1])
                mask_imgarray, shape = dcm_as_numpy_array(filepath_dcm)
                if shape == None:
                    continue
                if shape[0] < 4200:
                    filepath_dcm = join(croppath, dicom_imgname[0])
                    mask_imgarray, shape = dcm_as_numpy_array(filepath_dcm)
                if shape == None or shape[0] < 4200:
                    continue
                print(shape)
                if posf == 'MALIGNANT':
                    pos = True
                elif posf == 'BENIGN':
                    pos = False
                else:
                    print('BENIGN_WITHOUT_CALLBACK')
                    continue
                sample_patches(full_imgarray, mask_imgarray, save_dir, filename, int(abn), pos, 
                               pos_cutoff=0.8, neg_cutoff=0.3, nb_bkg=3, nb_abn=10, itype=itype)
                print('done.')
                