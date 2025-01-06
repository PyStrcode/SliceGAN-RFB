import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from slicegan import ASPPF_MSG
import torch.nn.utils.spectral_norm as spectral_norm
def slicegan_nets(pth, Project_name, Training, imtype, dk, ds, df, dp, gk, gs, gf, gp):
    """
    Define a generator and Discriminator
    :param Training: If training, we save params, if not, we load params from previous.
    This keeps the parameters consistent for older models
    :return:
    """
    #save params
    # pth = pth + '/' + Project_name
    params = [dk, ds, df, dp, gk, gs, gf, gp]
    # if fresh training, save params
    if Training:
        pth = pth + '/' + Project_name
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    # if loading model, load the associated params file
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp = pickle.load(filehandle)


    # Make nets
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for lay, (k,s,p) in enumerate(zip(gk,gs,gp)):
                self.convs.append(nn.ConvTranspose3d(gf[lay], gf[lay+1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay+1]))

        def forward(self, x):
            for conv,bn in zip(self.convs[:-1],self.bns[:-1]):
                x = F.relu_(bn(conv(x)))
            #use tanh if colour or grayscale, otherwise softmax for one hot encoded
            if imtype in ['grayscale', 'colour']:
                out = 0.5*(torch.tanh(self.convs[-1](x))+1)
            else:
                out = torch.softmax(self.convs[-1](x),1)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))

        def forward(self, x):
            for conv in self.convs[:-1]:
                x = F.relu_(conv(x))
            x = self.convs[-1](x)
            return x

    return Discriminator, Generator
def slicegan_rc_nets(pth, Project_name, Training, imtype, img_channels, dk,ds,df,dp,gk,gs,gf,gp):
    """
    Define a generator and Discriminator
    :param Training: If training, we save params, if not, we load params from previous.
    This keeps the parameters consistent for older models
    :return:
    """
    #save params
    # pth = pth + '/' + Project_name
    params = [dk, ds, df, dp, gk, gs, gf, gp]
    # if fresh training, save params
    if Training:
        # with open(pth + '/*.png', 'wb') as filehandle:
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    # if loading model, load the associated params file
    else:
        # with open(pth + '/*.png', 'rb') as filehandle:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp = pickle.load(filehandle)


    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            # self.asppG = ASPPF.BasicRFB(6, 6)
            self.rcconv = nn.Conv3d(gf[-2],gf[-1],3,1,0)
            for lay, (k,s,p) in enumerate(zip(gk,gs,gp)):
                self.convs.append(nn.ConvTranspose3d(gf[lay], gf[lay+1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay+1]))

        def forward(self, x):
            for lay, (conv, bn) in enumerate(zip(self.convs[:-1],self.bns[:-1])):
                x = F.relu_(bn(conv(x)))
            size = (int(x.shape[2]-1,)*2,int(x.shape[3]-1,)*2,int(x.shape[3]-1,)*2)
            up = nn.Upsample(size=size, mode='trilinear', align_corners=False)
            out = torch.softmax(self.rcconv(up(x)), 1)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            # self.BN = nn.ModuleList()
            self.asppD = ASPPF_MSG.BasicRFB(img_channels, img_channels).cuda()
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                conv = nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False)
                self.convs.append(conv)

        def forward(self, x):
            x_list = self.asppD(x)
            x_list.append(x)

            for i in range(len(x_list)):
                x_ = x_list[i]
                for conv in self.convs[:-1]:
                    x_ = F.relu_(conv(x_))
                if i == 0:
                    x_fix = self.convs[-1](x_)
                else:
                    x_fix = self.convs[-1](x_) + x_fix
            return x_fix / len(x_list)

    return Discriminator, Generator
    
