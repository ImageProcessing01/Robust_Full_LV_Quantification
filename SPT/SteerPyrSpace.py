import numpy as np
from SPT.SteerPyrUtils import sp5_filters, sp3_filters, sp1_filters, sp0_filters
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision import utils as vutil
from PIL import Image

def steer(basis, angle, harmonics, steermtx):
    steervect = np.zeros((1, harmonics.shape[0]*2))
    arg = angle * harmonics
    for i in range(0, harmonics.shape[0]):
        steervect[0, i*2] = np.cos(arg[i])
        steervect[0, i*2+1] = np.sin(arg[i])
    
    steervect = np.dot(steervect, steermtx)
    for i in range(0, harmonics.shape[0]*2):
        basis[i] = basis[i] * steervect[0, i]
        
    res = basis[0]
    for i in range(1, harmonics.shape[0]*2):
        res = res + basis[i]    
    return res

def corrDn(image, filt, step=1, channels=1):
    filt_ = torch.from_numpy(filt).float().unsqueeze(0).unsqueeze(0).repeat(channels,1,1,1).to(image.device)
    p = (filt_.shape[2]-1)//2
    image = F.pad(image, (p,p,p,p),'reflect')
    img = F.conv2d(image, filt_, stride=step, padding=0, groups = channels)
    return img

def SteerablePyramidSpace(image, height=3, order=5, channels=1, filter_name='sp3_filters'):
    num_orientations = order + 1
    if filter_name == 'sp0_filters':
        filters = sp0_filters()
    elif filter_name == 'sp1_filters':
        filters = sp1_filters()
    elif filter_name == 'sp3_filters':
        filters = sp3_filters()
    elif filter_name == 'sp5_filters':
        filters = sp5_filters()
    harmonics, steermtx = filters['harmonics'], filters['mtx']
   
    hi0 = corrDn(image, filters['hi0filt'], step=1, channels=channels)
    pyr_coeffs = []
    pyr_coeffs.append(hi0)
    lo = corrDn(image, filters['lo0filt'], step=1, channels=channels)
    l_s0 = lo
    for _ in range(height):
        bfiltsz = int(np.floor(np.sqrt(filters['bfilts'].shape[0])))
        for b in range(num_orientations):
            filt = filters['bfilts'][:, b].reshape(bfiltsz, bfiltsz).T
            band = corrDn(lo, filt, step=1, channels=channels)
            pyr_coeffs.append(band)
        lo = corrDn(lo, filters['lofilt'], step=2, channels=channels)

    pyr_coeffs.append(lo)
    pyr_coeffs.append(l_s0)
    return pyr_coeffs, harmonics, steermtx

def filter_three(img, hei=2, imgSize=(80,80)):
    max_size = img.shape[-1]
    #print(max_size)
    y, harmonics, steermtx = SteerablePyramidSpace(img, height=hei, order=3, 
                                                   channels=1, filter_name='sp3_filters')
    s1, s2 = y[1:5], y[5:9]
    s1d1 = steer(s1, np.pi/6, harmonics, steermtx)
    s1d2 = steer(s1, np.pi*3/6, harmonics, steermtx)
    s1d3 = steer(s1, np.pi*5/6, harmonics, steermtx)
    
    s2d1 = steer(s2, np.pi/6, harmonics, steermtx)
    s2d2 = steer(s2, np.pi*3/6, harmonics, steermtx)
    s2d3 = steer(s2, np.pi*5/6, harmonics, steermtx)
    
    s1d1, s1d2 = s1d1.squeeze().unsqueeze(1), s1d2.squeeze().unsqueeze(1)
    s1d3 = s1d3.squeeze().unsqueeze(1)
    s2d1, s2d2 = s2d1.squeeze().unsqueeze(1), s2d2.squeeze().unsqueeze(1)
    s2d3 = s2d3.squeeze().unsqueeze(1)
    
    s2d1 = F.interpolate(s2d1, size=[max_size, max_size], mode='bilinear')
    s2d2 = F.interpolate(s2d2, size=[max_size, max_size], mode='bilinear')
    s2d3 = F.interpolate(s2d3, size=[max_size, max_size], mode='bilinear')
    
    s3 = y[9].squeeze().unsqueeze(1)
    s3 = F.interpolate(s3, size=[max_size, max_size], mode='bilinear')

    return s1d1, s1d2, s1d3, s2d1, s2d2, s2d3, s3

def getSPT(img, imgSize=(80,80), device=torch.device("cuda")):
    
    x = img
    x = x.to(device)
    x.requires_grad_(True)
    
    s1d1_3, s1d2_3, s1d3_3, s2d1_3, s2d2_3, s2d3_3, s3_3 = filter_three(x)
    
    d1 = torch.cat((s1d1_3, s2d1_3, s3_3), 1)
    d2 = torch.cat((s1d2_3, s2d2_3, s3_3), 1)
    d3 = torch.cat((s1d3_3, s2d3_3, s3_3), 1)
    return d1, d2, d3




