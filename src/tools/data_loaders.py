'''
AHAAB data_loaders submodule
Part of the AHAAB tools module

ahaab/src/
└──tools/
    └──data_loaders.py

Submodule list:
    
    AhaabAtomDataset
'''

# PyTorch
import torch
from torch.utils.data import Dataset
from torch import tensor

# Python base libraries

class AhaabAtomDataset(Dataset):
    '''
    A custom dataset class that inherits the properties of pytorch Dataset.

    Usage:

    Dataset=AhaabAtomDataset(feature_data,pkd)
    > feature_data: pandas dataframe containing feature-level data
    > pkd: pandas dataframe with a single column, where each value corresponds
      to a label to the corresponding row in feature_data
    '''
    # Custom dataset class (inherets pytorch Dataset). Takes in two dataframes: one of feature data and one of pkd values. Returns a pytroch Dataset object that can then be loaded into a DataLoader. 

    # Feature data includes 
    
    def __init__(self, feature_data, pkd):
        self.x=tensor(feature_data.values,dtype=torch.float32)
        self.y=tensor(pkd.values,dtype=torch.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,index):
        return self.x[index], self.y[index]