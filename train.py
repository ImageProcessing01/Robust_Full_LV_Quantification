import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import time
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import *
from config import params
from datasets.cardiac_dig import CardiacDigProvider
from datasets.MMdataset import MMDatasetProvider
from torch.utils.tensorboard import SummaryWriter

if(params['dataset'] == 'cardiac_dig'):
    from models.CardiacDig_model import RobustNet
elif(params['dataset'] == 'MMdataset'):
    from models.MMdataset_model import RobustNet

# Set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Use GPU if available.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

if(params['dataset'] == 'cardiac_dig'):
    dataset = CardiacDigProvider(params['cross_valid'])
elif(params['dataset'] == 'MMdataset'):
    dataset = MMDatasetProvider()
train_loader, test_loader, valid_loader = dataset.train, dataset.test, dataset.valid

model = RobustNet().to(device)

pretrained_path = 'best.pt'
if os.path.exists(pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    model.load_state_dict(pretrained_dict['model'])
    model.task_cov_var.data = torch.tensor(pretrained_dict['task']).to(device)
    model.class_cov_var.data = torch.tensor(pretrained_dict['class']).to(device)
    model.feature_cov_var.data = torch.tensor(pretrained_dict['feature']).to(device)
else:
    model.apply(weights_init)

criterion = TensorNormalLoss()
criterion_test = L1TestLoss()

# Adam optimiser is used.
optim = optim.Adam([{'params': model.parameters()}], lr=params['lr'], betas=(params['beta1'], params['beta2']))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, 
                     milestones=params['multistep'], gamma=params['lr_gamma'])

print("-"*25)
print("Starting Training Loop...\n")
print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(params['dataset']) % (params['num_epochs'], params['batch_size'], len(train_loader)))
print("-"*25)

for epoch in range(params['num_epochs']):
    model.train()
    avg_loss, avg_mtloss, count = 0.0, 0., 0
    for i, data in enumerate(train_loader):
        optim.zero_grad()
        img, label, d1annot, d2annot, d3annot, pix = data
        #print(img.shape)
        imgin = img.to(torch.float32).to(device)
        outd1, outd2, outd3, mtloss = model(imgin)
        #print(out1.size())
        label = label.to(device)
        d1annot, d2annot, d3annot = d1annot.to(device), d2annot.to(device), d3annot.to(device)
        loss, mtloss = criterion(outd1, outd2, outd3, label, mtloss, d1annot, d2annot, d3annot)
        print("epoch: " + str(epoch) + "   loss: " + str(loss.item()) )
        loss.backward()
        optim.step()
        avg_loss += loss.item() - mtloss.item()*0.001
        avg_mtloss += mtloss.item()*0.001
        count += 1
    with SummaryWriter('runs/{}_{}/'.format(params['dataset'], params['lr'],)) as writer:
        writer.add_scalar('train_loss', avg_loss/count, epoch)
        writer.add_scalar('train_mtloss', avg_mtloss/count, epoch)
    scheduler.step()
    
    all_loss = np.zeros((1, 11))
    for i, data in enumerate(test_loader):
        img, label, d1annot, d2annot, d3annot, pix = data
        model.eval()
        imgin = img.to(torch.float32).to(device)
        outd1, outd2, outd3, _ = model(imgin)
        label = label.to(device)
        d1annot, d2annot, d3annot = d1annot.to(device), d2annot.to(device), d3annot.to(device)
        loss, out, label = criterion_test(outd1, outd2, outd3, label, d1annot, d2annot, d3annot)
        
        loss = loss.squeeze().detach().cpu().numpy()
        pix = torch.reshape(pix, (-1, 1)).numpy()
        #print(pix)
        loss = np.array(loss)
        for cc in range(0, 2):
            for dd in range(0, loss.shape[0]):
                loss[dd, cc] = loss[dd, cc] * pix[0,0] * pix[0,0] *120*120.0
        for cc in range(2, 11):
            for dd in range(0, loss.shape[0]):
                loss[dd, cc] = loss[dd, cc] * pix[0,0] * 120.0
        for kk in range(loss.shape[0]): 
            all_loss = np.insert(all_loss, 0, values=loss[kk, :], axis=0)
    
    all_loss = np.delete(all_loss, -1, axis=0)
    m_loss = np.mean(all_loss, axis=0)
    s_loss = np.std(all_loss, axis=0)
    
    with SummaryWriter('runs/{}_{}/'.format(params['dataset'], params['lr'],)) as writer:
        writer.add_scalar('test_area', np.mean(m_loss[0:2]), epoch)
        writer.add_scalar('test_dim', np.mean(m_loss[2:5]), epoch)
        writer.add_scalar('test_rwt', np.mean(m_loss[5:]), epoch)
        print(np.mean(m_loss[0:2]), np.mean(m_loss[2:5]), np.mean(m_loss[5:]))
    
    
    if epoch%3==2:
        model.update_cov()
          
    if (epoch+1) < 150:
        torch.save({
            'class' : model.class_cov_var.detach().cpu().numpy(),
            'task' : model.task_cov_var.detach().cpu().numpy(),
            'feature' : model.feature_cov_var.detach().cpu().numpy(),
            'model' : model.state_dict(),
            'optim' : optim.state_dict(),
            }, 'checkpoint/model_epoch_%d_{}.pt'.format(params['dataset']) %(epoch+1))




