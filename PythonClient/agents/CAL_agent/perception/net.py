import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torchvision import models
from numpy.lib.stride_tricks import as_strided
from ipdb import set_trace

### helper functions

def tile_array(a, b0, b1):
    r, c = a.shape                                    # number of rows/columns
    rs, cs = a.strides                                # row/column strides
    x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0)) # view a as larger 4D array
    return x.reshape(r*b0, c*b1)                      # create new 2D array

def get_bool_vec(d, n_h):
    # first one hot encode to three possible classes
    t = np.zeros((len(d), 3))
    t[np.arange(len(d)), d+1] = 1
    # upscale to correct hidden_sz
    bool_vec = tile_array(t, 1, n_h//3)
    return torch.Tensor(bool_vec)

### custom modules

class Flatten(nn.Module):
    def __init__(self, full=False):
        super(Flatten,self).__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)

class Swap1and2(nn.Module):
    def __init__(self): super(Swap1and2,self).__init__()

    def forward(self, x):
      print(x.size()) 
      return x.permute(0, 2, 1)

class TimeSlice(nn.Module):
    """
    given sequential input, return only sequence of length l with dilation d
    this module assumes that the time dim is the second dim (after batch dim)
    """
    def __init__(self, l, d=1):
        super(TimeSlice,self).__init__()
        self.l = l
        self.d = d

    def forward(self, x):
        print('Before TimeSlice"',x.size())
        if self.d>1: # apply dilation
            x = torch.flip(x, dims=[1])
            x = x[:, ::self.d]
            x = torch.flip(x, dims=[1])
        print('L:',self.l)
        if self.l==-1: # return full sequence
            return x
        else:
            print(x.size())
            return x[:, -self.l:]
### params

class NetworkParams():
    name = 'default'
    type_ = 'LSTM'
    p = 0.27
    n_h = 100
    lr = 1e-4
    dil = 1
    seq_len = 10

    def update(self, params):
        if params is None:
            return True

        for k, v in params.items():
                setattr(self, k, v)

### main network

class TaskBlock(nn.Module):
    def __init__(self, params, n_in, n_out, cond=False):
        super(TaskBlock, self).__init__()
        self.cond = cond
        self.n_h = params.n_h if not cond else params.n_h*3
        self.type = params.type_
        self.discrete = n_out > 1
        self.seq_len = params.seq_len

        self.bn = nn.BatchNorm1d(self.n_h)
        self.dropout = nn.Dropout(params.p)
        self.lin_out = nn.Linear(self.n_h, n_out)

        # different core units dependent on task block type
        if params.type_=='MLP':
            self.core = nn.Sequential(
                TimeSlice(1),
                Flatten(),
                nn.Linear(n_in, self.n_h),
                nn.ReLU(inplace=False),
            )
        elif params.type_=='LSTM':
            self.core = nn.Sequential(
                TimeSlice(l=params.seq_len, d=params.dil),
                nn.LSTM(n_in, self.n_h, 1, batch_first=True),
            )
        elif params.type_=='GRU':
            self.core = nn.Sequential(
                TimeSlice(l=params.seq_len, d=params.dil),
                nn.GRU(n_in, self.n_h, 1, batch_first=True),
            )
        elif params.type_=='TempConv':
            with open('temp_seq.txt','a') as ts:
                ts.write(str(params.seq_len)+'\n')

            print("x[1]", str(params))
            print("TempConv initiail", )

            print('Sequence Length of tempconv',str(params.seq_len))
            self.core = nn.Sequential(
                TimeSlice(l=self.seq_len, d=params.dil),
                Swap1and2(),
                nn.Conv1d(n_in, self.n_h, self.seq_len),
                nn.ReLU(inplace=False),
            )
        else:
            raise ValueError("Task block type not known")

    def forward(self, x_in, d=None):
        print("it's in")
        print(self.type)
        print(self.seq_len)
        print('size',x_in.size())
        x = self.core(x_in)
        if self.type=='LSTM':
            x = x[1][1]
            print(x.size())
        elif self.type=='GRU':
            print('Initial',x[0].size())
            x = x[1]
            print(x.size())
        elif self.type=='TempConv':
            #print("x[1]", x[1])
            #print("TempConv initiail")
            x = x.squeeze()

        #if self.discrete: x = self.bn(x)
        x = self.dropout(x)

        # handle conditional affordances
        if self.cond:
            bool_vec = get_bool_vec(d, self.n_h).to(x.device)
            x = x.reshape(1,x.size()[0])
            print('Bool vector',bool_vec.size())
            print('x vector',x.size())
            x *= bool_vec

        x = self.lin_out(x)
        # # handle discrete affordances
        return x

class CAL_network(nn.Module):
    def __init__(self, p):
        super(CAL_network, self).__init__()
        #self.params = p

        # get feature extractor and first FCN layer from vgg
        vgg = models.vgg11_bn(pretrained=True)
        ls = [l for l in vgg.features]+ [nn.AdaptiveMaxPool2d(1), Flatten()]
        self.features = nn.Sequential(*ls)
        n_in = 512 # fixed amount of features after feature extractor

        # initialize the task blocks
        p.type_='GRU'
        p.seq_len=14#10
        p.dil=2
        p.n_h=120
        p.p=0.27
        self.red_light = TaskBlock(params=p, n_in=512, n_out=2)
        p.type_='TempConv'
        p.seq_len=6#10
        p.dil=1
        p.n_h=120
        p.p=.68
        self.hazard_stop = TaskBlock(params=p, n_in=512, n_out=2)
        p.type_='MLP'
        p.seq_len=1#10
        p.n_h=100
        p.p=0.55
        self.speed_sign = TaskBlock(params=p, n_in=512,  n_out=4)
        p.type_='GRU'
        p.seq_len=11#10
        p.n_h=120
        p.p=0.38
        self.veh_distance = TaskBlock(params=p, n_in=512, n_out=1)
        p.type_='TempConv'
        p.seq_len=10
        p.n_h=100
        p.p=0.44
        self.relative_angle = TaskBlock(params=p, n_in=512, n_out=1, cond=True)
        p.type_='TempConv'
        p.seq_len=10
        p.n_h=100
        p.p=0.44
        self.center_distance = TaskBlock(params=p, n_in=512, n_out=1, cond=True)

    def forward(self, inputs):
        # feature extractor over sequence, reshape appropriately
        x_in = inputs['sequence']
        print('Sequence size',x_in.size())
        batch_size,timesteps, C, H, W = x_in.size()

        c_in = x_in.view(batch_size * timesteps, C, H, W)
        c_out = self.features(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)

        # unconditional affordances
        pred = {}
        pred['red_light'] = self.red_light(r_in)
        pred['hazard_stop'] = self.hazard_stop(r_in)
        pred['speed_sign'] = self.speed_sign(r_in)
        pred['veh_distance'] = self.veh_distance(r_in)
        pred['relative_angle'] = self.relative_angle(r_in, inputs['direction'])
        pred['center_distance'] = self.center_distance(r_in, inputs['direction'])

        return pred

def get_model(params={}):
    p = NetworkParams()
    p.update(params)
    model = CAL_network(p)

    for name,param in model.named_parameters():
        if 'features' in name:
            param.requires_grad = False

    opt = optim.Adam(model.parameters(), lr=p.lr)
    return model, opt
