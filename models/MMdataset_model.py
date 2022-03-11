import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Process
#from sqrtm import sqrtm
from SPT import SteerPyrSpace
import MTlearn.tensor_op as tensor_op
import random

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

class L2Pooling(nn.Module):
    def __init__(self):
        super(L2Pooling, self).__init__()
        pass
    def forward(self, x):
        x = torch.mul(x, x)
        x = (torch.sum(torch.sum(x, -1), -1) + 1e-8) ** 0.5
        return x
    
class RobustNet(nn.Module):
    def __init__(self):
        super(RobustNet, self).__init__()
        
        self.bn1 = nn.Sequential(nn.BatchNorm2d(3),
                                 nn.LeakyReLU(inplace=True))
        
        self.bn2 = nn.Sequential(nn.BatchNorm2d(3),
                                 nn.LeakyReLU(inplace=True))
        
        self.bn3 = nn.Sequential(nn.BatchNorm2d(3),
                                  nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(nn.Conv2d(3, 60, kernel_size=7, stride=1, padding=3, dilation=1),
                                    nn.BatchNorm2d(60),
                                    nn.LeakyReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, padding=0),  # 60x60
                                   
                                    nn.Conv2d(60, 120, kernel_size=5, stride=1, padding=2, dilation=1),
                                    nn.BatchNorm2d(120),
                                    nn.LeakyReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, padding=0), #30
                                    
                                    nn.Conv2d(120, 240, kernel_size=5, stride=1, padding=2, dilation=1),
                                    nn.BatchNorm2d(240),
                                    nn.LeakyReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, padding=0),  # 15
                                    
                                    nn.Sequential(nn.Conv2d(240, 480, kernel_size=3, stride=1, padding=1, dilation=1),
                                    nn.BatchNorm2d(480),
                                    nn.LeakyReLU(inplace=True),   
                                    nn.MaxPool2d(kernel_size=2, padding=0)),# 7
                                    
                                    nn.Conv2d(480, 480, kernel_size=3, stride=1, padding=1, dilation=1),
                                    nn.BatchNorm2d(480),
                                    nn.LeakyReLU(inplace=True),
                                    
                                    L2Pooling()) #10x10
        
        self.fc = nn.Sequential(nn.Linear(480, 100),
                                nn.Dropout(0.3),
                                nn.LeakyReLU(inplace=True))

        self.lstm_share = nn.Sequential(nn.LSTM(100, 100, num_layers=1))
        
        self.fcd1_last = nn.Linear(100, 5)
        self.fcd2_last = nn.Linear(100, 5)
        self.fcd3_last = nn.Linear(100, 5)

        self.task_cov_var = Variable(torch.eye(3)).to(device)
        self.class_cov_var = Variable(torch.eye(5)).to(device)
        self.feature_cov_var = Variable(torch.eye(100)).to(device)
        
        self.noise = nn.LeakyReLU(inplace=True)
                
    def forward(self, x):

        batch, frm_slc = x.shape[0], x.shape[1]

        x = torch.reshape(x, (batch*frm_slc, 1, x.shape[-2], x.shape[-1]))
        d1, d2, d3 = SteerPyrSpace.getSPT(x)
        
        d1 = self.noise_f(d1)
        d2 = self.noise_f(d2)
        d3 = self.noise_f(d3)
        
        d1 = self.bn1(d1)
        d2 = self.bn2(d2)
        d3 = self.bn3(d3)
        
        d1 = self.conv(d1).view(-1, 480)
        d2 = self.conv(d2).view(-1, 480)
        d3 = self.conv(d3).view(-1, 480)
        
        d1 = self.fc(d1)
        d2 = self.fc(d2)
        d3 = self.fc(d3)
        
        d1 = torch.reshape(d1, (frm_slc, -1, 100))
        d2 = torch.reshape(d2, (frm_slc, -1, 100))
        d3 = torch.reshape(d3, (frm_slc, -1, 100))
        
        d1_lstm_out, (_, _) = self.lstm_share(d1)
        d2_lstm_out, (_, _) = self.lstm_share(d2)
        d3_lstm_out, (_, _) = self.lstm_share(d3)
        
        d1_last_in = torch.reshape(d1_lstm_out, (-1, 100))
        d2_last_in = torch.reshape(d2_lstm_out, (-1, 100))
        d3_last_in = torch.reshape(d3_lstm_out, (-1, 100))
        
        outd1 = self.fcd1_last(d1_last_in)
        outd2 = self.fcd2_last(d2_last_in)
        outd3 = self.fcd3_last(d3_last_in)
        
        mt_loss = self.multitask_loss()

        return outd1, outd2, outd3, mt_loss
    
    def noise_f(self, x, scale=0.07):
        pos = random.uniform(0, 16)
        if pos < 8:
            eps = 10e-5
            x *= 0.35
            x += 0.07
            x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=scale).rsample() * torch.sqrt(F.relu(x.clone()) + eps)
            x -= 0.07
            x /= 0.35      
        return self.noise(x)
    
    def select_func(self, x):
            if x > 0.1:
                return 1. / x
            else:
                return x
    
    def multitask_loss(self):
        wd1 = self.fcd1_last.weight.view(1, 5, 100)
        wd2 = self.fcd2_last.weight.view(1, 5, 100)
        wd3 = self.fcd3_last.weight.view(1, 5, 100)
        weights = torch.cat((wd1, wd2, wd3), dim=0).contiguous()
   
        multi_task_loss = tensor_op.MultiTaskLoss(weights, self.task_cov_var, self.class_cov_var, self.feature_cov_var)
        return multi_task_loss
    
    def update_cov(self):
        # get updated weights
        wd1 = self.fcd1_last.weight.view(1, 5, 100)
        wd2 = self.fcd2_last.weight.view(1, 5, 100)
        wd3 = self.fcd3_last.weight.view(1, 5, 100)
        weights = torch.cat((wd1, wd2, wd3), dim=0).contiguous()

        # update cov parameters
        temp_task_cov_var = tensor_op.UpdateCov(weights.data, self.class_cov_var.data, self.feature_cov_var.data)
        temp_class_cov_var = tensor_op.UpdateCov(weights.data.permute(1, 0, 2).contiguous(), self.task_cov_var.data, self.feature_cov_var.data)
        temp_feature_cov_var = tensor_op.UpdateCov(weights.data.permute(2, 0, 1).contiguous(), self.task_cov_var.data, self.class_cov_var.data)

        # task covariance
        u, s, v = torch.svd(temp_task_cov_var)
        s = s.cpu().apply_(self.select_func).to(device)
        self.task_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
        this_trace = torch.trace(self.task_cov_var)
        if this_trace > 3000.0:        
            self.task_cov_var = Variable(self.task_cov_var / this_trace * 3000.0).to(device)
        else:
            self.task_cov_var = Variable(self.task_cov_var).to(device)

        # class covariance
        u, s, v = torch.svd(temp_class_cov_var)
        s = s.cpu().apply_(self.select_func).to(device)
        self.class_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
        this_trace = torch.trace(self.class_cov_var)
        if this_trace > 3000.0:        
            self.class_cov_var = Variable(self.class_cov_var / this_trace * 3000.0).to(device)
        else:
            self.class_cov_var = Variable(self.class_cov_var).to(device)
        
        # feature covariance
        u, s, v = torch.svd(temp_feature_cov_var)
        s = s.cpu().apply_(self.select_func).to(device)
        temp_feature_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
        this_trace = torch.trace(temp_feature_cov_var)
        if this_trace > 1000.0:        
            self.feature_cov_var += 0.001 * Variable(temp_feature_cov_var / this_trace * 1000.0).to(device)
        else:
            self.feature_cov_var += 0.001 * Variable(temp_feature_cov_var).to(device)
