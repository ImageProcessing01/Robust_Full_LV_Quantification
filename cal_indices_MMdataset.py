import os
import time
import re
import math
import torch
import argparse
import numpy as np
import scipy.io as sio
import pandas as pd
from glob import glob
import nibabel as nib
from PIL import Image
import SimpleITK as sitk
from medpy.metric.binary import hd, dc

import matplotlib.pyplot as plt
import torchvision.utils as vutils

import cv2
from math import *


def cal_areas(mask, endo=3, myo=2, wid=80):
    mask = np.array(mask)
    if len(np.where(mask == 1)[0])==0:
        endo, myo = 1, 2
    
    myoc = np.where(mask == myo)
    endoc = np.where(mask == endo)

    epic = (len(myoc[0]) + len(endoc[0]) ) / wid / wid
    endoc = len(endoc[0]) / wid / wid
    
    return [epic, endoc]

def cal_rwt(mask, endo=3, myo=2, wid=80):
    mask1 = Image.fromarray(np.uint8(mask))
    if len(np.where(mask1 == 1)[0])==0:
        endo, myo = 1, 2
    line = []
    for a in range(0, 120):
        mask2 = mask1.rotate(-3*a)
        mask3 = np.array(mask2)
        ind1 = np.where(mask3  == endo)
        cx = int((np.max(ind1[1])+np.min(ind1[1]))/2)
        cy = int((np.max(ind1[0])+np.min(ind1[0]))/2)
        
        x, y = cx, cy
        ll = []
        while True:
            if y>wid-1 or y<0 or x<0 or x>wid-1:
                break
            mask3 = np.array(mask2, dtype='float32')
            if mask3[y, x]==myo:
                ll.append([y, x])
            x += 1
        line.append(ll)

    line_len = []
    for l in line:
        line_len.append(len(l))
    
    line_len = np.array(line_len)
    t1 = np.sum(line_len[60:80])/20/wid
    t2 = np.sum(line_len[80:100])/20/wid
    t3 = np.sum(line_len[100:120])/20/wid
    t4 = np.sum(line_len[0:20])/20/wid
    t5 = np.sum(line_len[20:40])/20/wid
    t6 = np.sum(line_len[40:60])/20/wid
    
    return [t1, t2, t3, t4, t5, t6]

def cal_dim(mask, endo=3, myo=2, wid=80):
    mask1 = Image.fromarray(np.uint8(mask))
    if len(np.where(mask1 == 1)[0])==0:
        endo, myo = 1, 2
    
    line = []
    for a in [-20,-25,-26,-27,-28,-29,-30,-31,-32,-33,-34,-35,-40,
              80,85,86,87,88,89,90,91,92,93,94,95,100,
              20,25,26,27,28,29,30,31,32,33,34,35,40, ]:
        mask2 = mask1.rotate(a)
        mask3 = np.array(mask2)
        ind1 = np.where(mask3  == endo)
        cy = int((np.max(ind1[0])+np.min(ind1[0]))/2)
        
        x, y = 0, cy
        ll = []
        while True:
            if y>wid-1 or y<0 or x<0 or x>wid-1:
                break
            mask3 = np.array(mask2, dtype='float32')
            if mask3[y, x]==endo:
                ll.append([y, x])
            x += 1
        line.append(ll)

    line_len = []
    for l in line:
        line_len.append(len(l))
    
    line_len = np.array(line_len)
    d1 = np.mean(line_len[0:13])/wid
    d2 = np.mean(line_len[13:26])/wid
    d3 = np.mean(line_len[26:39])/wid
    
    return [d1, d2, d3]


def cal_angle(gt_frame):
    
    rv_cx, rv_cy, myo_cx, myo_cy, count1, count2 = 0, 0, 0, 0, 0, 0
    for m in range(0, gt_frame.shape[2]):
        ind3 = np.where(gt_frame[:, :, m]==3)
        if len(ind3[0])>0:
            rv_cy += np.mean(ind3[0])
            rv_cx += np.mean(ind3[1])
            count1 += 1
        
        ind2 = np.where(gt_frame[:, :, m]==2)
        if len(ind2[0])>0:
            myo_cy += np.mean(ind2[0])
            myo_cx += np.mean(ind2[1])
            count2 += 1
    
    rv_cx, rv_cy = int(rv_cx/count1), int(gt_frame.shape[0]-1-rv_cy/count1)
    myo_cx, myo_cy = int(myo_cx/count2), int(gt_frame.shape[0]-1-myo_cy/count2)
    
    if rv_cy-myo_cy>=0 and rv_cx-myo_cx>0:
        angle = 180 - math.degrees(math.atan((rv_cy-myo_cy) / (rv_cx-myo_cx)))
    elif rv_cy-myo_cy>=0 and rv_cx-myo_cx<0:
        angle = 1 * math.degrees(math.atan((rv_cy-myo_cy) / math.fabs(rv_cx-myo_cx)))
    elif rv_cy-myo_cy<=0 and rv_cx-myo_cx<0:
        angle = -1 * math.degrees(math.atan(math.fabs(rv_cy-myo_cy) / math.fabs(rv_cx-myo_cx)))
    elif rv_cy-myo_cy<=0 and rv_cx-myo_cx>0:
        angle = 180 + math.degrees(math.atan(math.fabs(rv_cy-myo_cy) / math.fabs(rv_cx-myo_cx)))
    elif rv_cy-myo_cy>0 and rv_cx-myo_cx==0:
        angle = 90
    elif rv_cy-myo_cy<0 and rv_cx-myo_cx==0:
        angle = -90
    return angle

def cal_LV_patch(gt_frame, img_frame):
    ind1 = np.where(gt_frame == 1)
    endo_cy = np.mean(ind1[1])
    endo_cx = np.mean(ind1[2])
    
    ind2 = np.where(gt_frame == 2)
    ymax, xmax = np.max(ind2[1]), np.max(ind2[2])
    ymin, xmin = np.min(ind2[1]), np.min(ind2[2])
    yrange = np.array([int(ymin-(ymax-ymin)/4), int(ymax+(ymax-ymin)/4)])
    
    yrange = np.clip(yrange, 0, gt_frame.shape[1]-1)
    
    myo_cy = np.mean(ind2[1])
    myo_cx = np.mean(ind2[2])
    
    cx = 0.8*endo_cx + 0.2*myo_cx
    cy = 0.8*endo_cy + 0.2*myo_cy
    
    xrange = np.array([int(cx-(yrange[1]-yrange[0])/2), int(cx+(yrange[1]-yrange[0])/2)]) - int((yrange[1]-yrange[0])/15)
    LV_patch = gt_frame[:, yrange[0]:yrange[1], xrange[0]:xrange[1]]
    img_patch = img_frame[:, yrange[0]:yrange[1], xrange[0]:xrange[1]]
    return LV_patch, img_patch
    
    
def rotate_without_loss(mris, degree):
    
    imgRs = []
    for i in range(0, mris.shape[2]):
        mri = mris[:, :, i]
        ind = np.where(mri>=1)
        
        img = (mri-np.min(mri)) / (np.max(mri)-np.min(mri))
        img = np.uint8(img*255.0)
        
        height, width = img.shape[:2]
        
        heightNew = int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
        widthNew = int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))
        matRotation = cv2.getRotationMatrix2D((width/2, height/2), degree, 1)
         
        matRotation[0,2] += (widthNew-width)/2   # 重点在这步，目前不懂为什么加这步
        matRotation[1,2] += (heightNew-height)/2 # 重点在这步
        imgRotation=cv2.warpAffine(img,matRotation,(widthNew,heightNew),borderValue=(0,0,0))
        
        imgR = np.float32(imgRotation)
        imgR = np.round(imgR / 255.0 * (np.max(mri)-np.min(mri)) + np.min(mri))
        imgRs.append(imgR)
    imgRs = np.array(imgRs)
    return imgRs
    
    
def get_MM_indices():
    
    path = 'OpenDataset/Testing/*/*_sa_gt.nii.gz'
    gt_paths = glob(path)
    
    count1, count2 = 0, 0
    for gt_path in gt_paths:
        ngt = nib.load(gt_path).get_fdata()
        ngt = np.array(ngt, dtype='float32')
        
        img_path = gt_path.replace('sa_gt.nii', 'sa.nii')
        nimg = nib.load(img_path).get_fdata()
        nimg = np.array(nimg, dtype='float32')
        
        for i in range(ngt.shape[3]):
            img_slcs = nimg[:, :, :, i]
            gt_slcs = ngt[:, :, :, i]
            ind1 = np.where(gt_slcs==1)
            
            slice_num = 0
            images_LV, myo_LV = [], []
            areas, dims, rwts = [], [], []
            if len(ind1[0])>0:
                try:
                    flair_file = img_path
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(flair_file)
                    
                    reader.LoadPrivateTagsOn()
                    reader.ReadImageInformation()
                    pix_spa = reader.GetMetaData('pixdim[1]')
                except:
                    pix_spa = 1.51
                
                angle = cal_angle(gt_slcs)
                gt_slcs = rotate_without_loss(gt_slcs, angle)
                img_slcs = rotate_without_loss(img_slcs, angle)
                
                gts_patch, imgs_patch = cal_LV_patch(gt_slcs, img_slcs)
                
                for j in range(imgs_patch.shape[0]):
                    gt_slc = gts_patch[j, :, :]
                    ind2 = np.where(gt_slc==2)
                    if len(ind2[0])>0:
                        
                        img_patch = imgs_patch[j, :, :]
                        gt_patch = gts_patch[j, :, :]
                        
                        areas.append(cal_areas(gt_patch, endo=1, myo=2, wid=gt_slc.shape[1]))
                        dims.append(cal_dim(gt_patch, endo=1, myo=2, wid=gt_slc.shape[1]))
                        rwts.append(cal_rwt(gt_patch, endo=1, myo=2, wid=gt_slc.shape[1]))
                        
                        img_patch = (img_patch-np.min(img_patch)) / (np.max(img_patch)-np.min(img_patch))
                        gt_patch0 = (gt_patch-np.min(gt_patch)) / (np.max(gt_patch)-np.min(gt_patch))
                        
                        im = Image.fromarray(np.uint8(img_patch*255.0))
                        im.save("MM_dataset/{}.png".format(count1))
                        img_lv = np.array(im, dtype='float32') / 255.0
                        images_LV.append(img_lv)
                        
                        gt_visual = Image.fromarray(np.uint8(gt_patch0*255.0))
                        gt_visual.save("MM_dataset/{}_gt.png".format(count1))
                        gt = Image.fromarray(np.uint8(gt_patch))
                        myo_lv = np.array(gt_patch, dtype='float32')
                        myo_LV.append(myo_lv)
                        
                        count1 += 1
                        
                
                count2 += 1
                if slice_num>-1:
                    images_LV = np.array(images_LV)
                    myo_LV = np.array(myo_LV)

                    areas, dims, rwts = np.array(areas), np.array(dims), np.array(rwts)
                    if np.mean(areas[1:3,1])<(areas[-1,1]+areas[-2,1])/2:
                        areas = areas[::-1, :].copy()
                        dims = dims[::-1, :].copy()
                        rwts = rwts[::-1, :].copy()
                        myo_LV = myo_LV[::-1, :, :].copy()
                        images_LV = images_LV[::-1, :, :].copy()
                    #print(images_LV.shape)
                    sio.savemat('MM_dataset/{}_ob.mat'.format(count2), {'images_LV': images_LV,
                                'myo_LV': myo_LV, 'areas':areas, 'dims':dims, 'rwt':rwts, 
                                'pix_spa':pix_spa})
                    
                    data = torch.Tensor(images_LV)
                    data = data.unsqueeze(1)
                    plt.figure(figsize=(20, 20))
                    plt.axis("off")
                    plt.imshow(np.transpose(vutils.make_grid(data, nrow=5, padding=5, normalize=True, pad_value=1), (1,2,0)))
                    plt.savefig("MM_dataset/{}_ob".format(count2))
                    plt.close('all')
                    
                    data = torch.Tensor(myo_LV)
                    data = data.unsqueeze(1)
                    plt.figure(figsize=(20, 20))
                    plt.axis("off")
                    plt.imshow(np.transpose(vutils.make_grid(data, nrow=5, padding=5, normalize=True, pad_value=1), (1,2,0)))
                    plt.savefig("MM_dataset/{}_ob_gt".format(count2))
                    plt.close('all')
                

if __name__ == "__main__":
    
    get_MM_indices()
   