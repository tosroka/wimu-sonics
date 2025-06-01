# taken from https://www.kaggle.com/code/rhythmcam/random-seed-everything
# basic random seed
import os 
import random
import numpy as np 

def seedBasic(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
# torch random seed
import torch
def seedTorch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seedEverything(seed):
    seedBasic(seed)
    seedTorch(seed)