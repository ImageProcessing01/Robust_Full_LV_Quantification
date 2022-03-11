import os
import torch
import torch.nn as nn
import numpy as np


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
    return

def weights_init(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            torch.nn.init.kaiming_normal_(layer.weight.data)    

class TensorNormalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outd1, outd2, outd3, label, mtloss, d1annot, d2annot, d3annot):
        
        label = torch.reshape(label, (-1, 11))
        d1label = torch.reshape(d1annot, (-1, 5))
        d2label = torch.reshape(d2annot, (-1, 5))
        d3label = torch.reshape(d3annot, (-1, 5))

        d1err = torch.abs(d1label[:, 0:3] - outd1[:, 0:3])
        d2err = torch.abs(d2label[:, 0:3] - outd2[:, 0:3])
        d3err = torch.abs(d3label[:, 0:3] - outd3[:, 0:3])
        area = (outd1[:, 3:] + outd2[:, 3:] + outd3[:, 3:]) / 3
        area_err = torch.abs(d1label[:, 3:]-area)

        err = torch.mean(torch.cat((d1err, d2err, d3err, area_err), 1))
        loss = err + 0.001*mtloss
        return loss, mtloss

class L1TestLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outd1, outd2, outd3, label, d1annot, d2annot, d3annot):
        
        d1annot, d2annot, d3annot = d1annot.squeeze(), d2annot.squeeze(), d3annot.squeeze()
        label = torch.reshape(label, (-1, 11))
        
        d1loss = torch.abs(d1annot - outd1)
        d2loss = torch.abs(d2annot - outd2)
        d3loss = torch.abs(d3annot - outd3)
        
        area = (outd1[:, 3:5] + outd2[:, 3:5] + outd3[:, 3:5])/3
        rwt = torch.cat((outd1[:, 1:3], outd2[:, 1:3], outd3[:, 1:3]), 1)
        dims = torch.cat((outd3[:, 0:1], outd1[:, 0:1], outd2[:, 0:1]), 1)
        
        out = torch.cat((area, dims, rwt), 1)
        
        arealoss = torch.abs(area - d1annot[:, 3:5])
        rwtloss = torch.cat((d1loss[:, 1:3], d2loss[:, 1:3], d3loss[:, 1:3]), 1)
        dimsloss = torch.cat((d1loss[:, 0:1], d2loss[:, 0:1], d3loss[:, 0:1]), 1)
        
        loss = torch.cat((arealoss, dimsloss, rwtloss), 1)
        
        return loss, out, label




