import numpy as np
import torch
import torch.nn.functional as F
import os, copy, time
#from tqdm import tqdm
import pandas as pd
from ipdb import set_trace
path = '/home/vasu/Desktop/project/'
### helper functions

def to_np(t):
    return np.array(t.cpu())

### Losses

def calc_class_weight(x, fac=2):
    """calculate inverse normalized count, multiply by given factor"""
    _, counts = np.unique(x, return_counts=True)
    tmp = 1/counts/sum(counts)
    tmp /= max(tmp)
    return tmp*fac

def get_class_weights():
    # class weights, calculated on the training set
    df_all = pd.read_csv(path + 'annotations.csv')
    return {
       'red_light': torch.Tensor(calc_class_weight(df_all['red_light'])),
       'hazard_stop': torch.Tensor(calc_class_weight(df_all['hazard_stop'])),
       'speed_sign': torch.Tensor(calc_class_weight(df_all['speed_sign'])),
       'relative_angle': torch.Tensor([1]),
       'center_distance': torch.Tensor([1]),
       'veh_distance': torch.Tensor([1]),
    }


w = get_class_weights()
print(w)
