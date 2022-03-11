import torch
import random
import numpy as np
import scipy.io as sio
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

train_range = [116, 87, 58, 29, 0]
valid_range = [101, 72, 43, 14, 130]
test_range = [116, 87, 58, 29, 0]

class CardiacDigDataset(data.Dataset):
    def __init__(self, dataset_path='cardiac-dig.mat', state='train', cross_valid_fold=0):

        assert (state in ['train', 'valid', 'test'])
        assert (cross_valid_fold in [0, 1, 2, 3, 4])
        self.state = state
        self.cvf = cross_valid_fold

        cardiac = sio.loadmat(dataset_path)
        self.images = cardiac['images_LV']
        self.areas = cardiac['areas']
        self.dims = cardiac['dims']
        self.rwts = cardiac['rwt']
        self.pix = np.array(sio.loadmat(annot_file)['pix_spa'])[0] * np.array(sio.loadmat(annot_file)['ratio'])[0]
        epi  = np.array(cardiac['epi_LV'], dtype='float32')
        endo = np.array(cardiac['endo_LV'], dtype='float32')
        self.myo = epi - endo

    def __getitem__(self, ind):

        index = ind
        if self.state=='train':
            if ind >= train_range[self.cvf]:
                index = ind + 29
        elif self.state=='valid' or self.state=='test':
            index = ind + valid_range[self.cvf]

        images = np.array(self.myo[:, :, index*20:index*20+20], dtype='float32').squeeze()
        dims = np.array(self.dims[:, index*20:index*20+20]).squeeze()
        areas = np.array(self.areas[:, index*20:index*20+20]).squeeze()
        rwts = np.array(self.rwts[:, index*20:index*20+20]).squeeze()
        annot = np.concatenate((areas, dims, rwts), axis=0)

        d1annot = np.array([annot[3,:], annot[5,:], annot[8,:], annot[0,:], annot[1,:]])
        d2annot = np.array([annot[4,:], annot[6,:], annot[9,:], annot[0,:], annot[1,:]])
        d3annot = np.array([annot[2,:], annot[7,:], annot[10,:], annot[0,:], annot[1,:]])

        images = images.transpose(2,0,1)
        annot = annot.transpose(1,0)
        print(d1annot.shape)
        d1annot, d2annot, d3annot = d1annot.transpose(1,0), d2annot.transpose(1,0), d3annot.transpose(1,0)
        
        if self.state=='train':
            images = DataAugmentation(images)

        return images, annot, d1annot, d2annot, d3annot, self.pix

    def __len__(self):
        if self.state=='train':
            return 101
        elif self.state=='valid':
            return 15
        elif self.state=='test':
            return 29

class CardiacDigProvider():

    def __init__(self, cross_valid_fold=0):
        
        dataset_path='data/cardiac-dig.mat'
        self.train = torch.utils.data.DataLoader(CardiacDigDataset(dataset_path, 'train', cross_valid_fold), 
                                                  batch_size=3, shuffle=True, 
                                                  num_workers=0,
                                                  worker_init_fn=seed_worker,
                                                  generator=g)
        
        self.valid = torch.utils.data.DataLoader(CardiacDigDataset(dataset_path, 'valid', cross_valid_fold), 
                                                  batch_size=1, shuffle=False, 
                                                  num_workers=0,
                                                  worker_init_fn=seed_worker,
                                                  generator=g)
        
        self.test = torch.utils.data.DataLoader(CardiacDigDataset(dataset_path, 'test', cross_valid_fold), 
                                                  batch_size=1, shuffle=False, 
                                                  num_workers=0,
                                                  worker_init_fn=seed_worker,
                                                  generator=g)

