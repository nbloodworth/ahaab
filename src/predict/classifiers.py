'''
AHAAB classifier submodule
Part of the AHAAB test module

ahaab/
└──predict
    └──classifiers.py

Submodule list:
    
    === AhaabAtomClassifier ===
'''

import torch.nn as nn

class AhaabAtomClassifier(nn.Module):
    '''
    AHAAB class defining the neural network
    for atom-based classification
    '''
    def __init__(self,feature_size=150,hidden_layer1=256,hidden_layer2=128):
        super().__init__()
        self.hidden1=nn.Linear(feature_size,hidden_layer1)
        self.act1=nn.ReLU()
        self.hidden2=nn.Linear(hidden_layer1,hidden_layer2)
        self.act2=nn.ReLU()
        self.output=nn.Linear(hidden_layer2,1)

    def forward(self,x):
        x=self.act1(self.hidden1(x))
        x=self.act2(self.hidden2(x))
        x=self.output(x)
        return x