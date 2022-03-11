import torch
import random
import numpy as np
import scipy.io as sio
from PIL import Image
import torch.utils.data as data

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from datasets.data_augment import *

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

class MMDataset(data.Dataset):
    def __init__(self, dataset_path='data/MMdataset/', state='train', length=150):
        
        assert (state in ['train', 'valid', 'test'])
        self.state = state
        self.path = dataset_path + state + '/'
        #self.path='img_corruption/gaussian_blur_0/'
        self.length = length
        
    def __getitem__(self, index):
        
        ind= index+1
        samp_path = self.path + '{}_ob.mat'.format(ind)
        cardiac = sio.loadmat(samp_path)
        images = cardiac['images_LV']
        areas = cardiac['areas']
        dims = cardiac['dims']
        rwts = cardiac['rwt']
        pix = np.array(float(cardiac['pix_spa'][0]))
        annot = np.concatenate((areas, dims, rwts), axis=1)

        d1annot = np.array([annot[:,4], annot[:,5], annot[:,8], annot[:,0], annot[:,1]])
        d2annot = np.array([annot[:,3], annot[:,6], annot[:,9], annot[:,0], annot[:,1]])
        d3annot = np.array([annot[:,2], annot[:,7], annot[:,10], annot[:,0], annot[:,1]])
        d1annot, d2annot, d3annot = d1annot.transpose(1,0), d2annot.transpose(1,0), d3annot.transpose(1,0)
        
        slice_num = images.shape[0]
        pix = pix * images.shape[-1] / 120.0
        images_resized = []
        for ii in range(images.shape[0]):
            img = Image.fromarray(np.uint8(images[ii, :, :] * 255.0))
            img = img.resize((120, 120))
            img = np.array(img, dtype='float32') / 255.
            images_resized.append(img)
        
        images_resized = np.array(images_resized)
        images_resized = DataAugmentation(images_resized)
        
        if self.state=='train':
            img_13slice = np.zeros((13, 120, 120))
            d1annot_13slice = np.zeros((13, 5))
            d2annot_13slice = np.zeros((13, 5))
            d3annot_13slice = np.zeros((13, 5))
            annot_13slice = np.zeros((13, 11))
            
            img_13slice[0:slice_num, :, :] = images_resized
            d1annot_13slice[0:slice_num, :] = d1annot
            d2annot_13slice[0:slice_num, :] = d2annot
            d3annot_13slice[0:slice_num, :] = d3annot
            annot_13slice[0:slice_num, :] = annot
            return img_13slice, annot_13slice, d1annot_13slice, d2annot_13slice, d3annot_13slice, pix
        else:
            return images_resized, annot, d1annot, d2annot, d3annot, pix

    def __len__(self):
        return self.length

class MMDatasetProvider():

    def __init__(self, cross_valid_fold=0): 
        
        dataset_path='data/MMdataset/'
        self.train = torch.utils.data.DataLoader(MMDataset(dataset_path, 'train', 300), 
                                                  batch_size=3, shuffle=True, 
                                                  num_workers=0,
                                                  worker_init_fn=seed_worker,
                                                  generator=g)
        
        self.valid = torch.utils.data.DataLoader(MMDataset(dataset_path, 'valid', 72), 
                                                  batch_size=1, shuffle=False, 
                                                  num_workers=0,
                                                  worker_init_fn=seed_worker,
                                                  generator=g)
        
        self.test = torch.utils.data.DataLoader(MMDataset(dataset_path, 'test', 270), 
                                                  batch_size=1, shuffle=False, 
                                                  num_workers=0,
                                                  worker_init_fn=seed_worker,
                                                  generator=g)


# if __name__=='__main__':
#     c = CardiacDigProvider()
#     dd = []
#     for i, d in enumerate(c.train):
#         print(d[0].shape)
#     dd = np.array(dd)
#     print(np.max(dd))

