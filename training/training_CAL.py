#!/usr/bin/env python
# coding: utf-8

# #### Setup

# In[1]:


# standard imports 
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim
from ipdb import set_trace

# jupyter setup
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# own modules
from dataloader import CAL_Dataset
from net import get_model
from dataloader import get_data, get_mini_data
from train import fit, custom_loss, validate
from metrics import calc_metrics

# paths
data_path = '/home/vasu/Desktop/project/CameraRGB/'


# uncomment the cell below if you want your experiments to yield always the same results

# In[2]:


# manualSeed = 42

# np.random.seed(manualSeed)
# torch.manual_seed(manualSeed)

# # if you are using GPU
# torch.cuda.manual_seed(manualSeed)
# torch.cuda.manual_seed_all(manualSeed)

# torch.backends.cudnn.enabled = False 
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


# #### Training

# Initialize the model. Possible Values for the task block type: MLP, LSTM, GRU, TempConv

# In[3]:


params = {'name': 'test', 'type_': 'MLP', 'lr': 3e-4, 'n_h': 128, 'p':0.5, 'seq_len':1}
model, opt = get_model(params)


# In[4]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# get the data loader. get mini data gets only a subset of the training data, on which we can try if the model is able to overfit

# In[5]:


train_dl, valid_dl = get_data(data_path, model.params.seq_len, batch_size=16)
# train_dl, valid_dl = get_mini_data(data_path, model.params.seq_len, batch_size=16, l=4000)


# uncomment the next cell if the feature extractor should also be trained

# In[6]:


# for name,param in model.named_parameters():
#     param.requires_grad = True
# opt = optim.Adam(model.parameters())


# Train the model. We automatically save the model with the lowest val_loss. If you want to continue the training and keep the loss history, just pass it as an additional argument as shown below.

# In[7]:


#!export CUDA_LAUNCH_BLOCKING = 1;


# In[8]:


model, val_hist = fit(10, model, custom_loss, opt, train_dl, valid_dl)


# In[9]:


# model, val_hist = fit(1, model, custom_loss, opt, train_dl, valid_dl, val_hist=val_hist)


# In[10]:


val_hist


# In[11]:


plt.plot(val_hist)


# #### evalute the model

# In[12]:


model, _ = get_model(params)
model.load_state_dict(torch.load(f"./models/{model.params.name}.pth"))
model.eval().to(device);


# In[13]:


model.eval();
_, all_preds, all_labels = validate(model, valid_dl, custom_loss)


# In[14]:


calc_metrics(all_preds, all_labels)


# #### plot results

# In[15]:


# for convience, we can pass an integer instead of the full string
int2key = {0: 'red_light', 1:'hazard_stop', 2:'speed_sign', 
           3:'relative_angle', 4: 'center_distance', 5: 'veh_distance'}


# In[16]:


def plot_preds(k, all_preds, all_labels, start=0, delta=1000):
    if isinstance(k, int): k = int2key[k]
    
    # get preds and labels
    class_labels = ['red_light', 'hazard_stop', 'speed_sign']
    pred = np.argmax(all_preds[k], axis=1) if k in class_labels else all_preds[k]
    label = all_labels[k][:, 1] if k in class_labels else all_labels[k]
    
    plt.plot(pred[start:start+delta], 'r--', label='Prediction', linewidth=2.0)
    plt.plot(label[start:start+delta], 'g', label='Ground Truth', linewidth=2.0)
    
    plt.legend()
    plt.grid()
    plt.show()


# In[17]:


plot_preds(5, all_preds, all_labels, start=0, delta=4000)


# In[ ]:




