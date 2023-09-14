'''
AHAAB classifier submodule
<<<<<<< HEAD
Part of the AHAAB predict module

ahaab/src/
└──predict/
=======
Part of the AHAAB test module

ahaab/
└──predict
>>>>>>> 4aaaf06f474a91f00483a2a78237897414a871db
    └──classifiers.py

Submodule list:
    
<<<<<<< HEAD
    AhaabAtomClassifierPKD
    AhaabAtomClassifierBinder
=======
    === AhaabAtomClassifier ===
>>>>>>> 4aaaf06f474a91f00483a2a78237897414a871db
'''

import torch.nn as nn

class AhaabAtomClassifierPKD(nn.Module):
    '''
<<<<<<< HEAD
    AHAAB class defining the neural network for atom-based classification 
    (regression classifier to predict pKd values).

    Usage:

    classifier=AhaabAtomClassifierPKD(feature_size)

    > Produces a neural network with input feature size defined by the input
      variable "feature size".
    '''
    def __init__(self,feature_size,hidden_layer1=512,hidden_layer2=256):
=======
    AHAAB class defining the neural network
    for atom-based classification (regression classifier to predict pKd values)
    '''
    def __init__(self,feature_size=140,hidden_layer1=256,hidden_layer2=128):
>>>>>>> 4aaaf06f474a91f00483a2a78237897414a871db
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

class AhaabAtomClassifierBinder(nn.Module):
    '''
    AHAAB class defining the neural network for discriminating binders
<<<<<<< HEAD
    from non-binders (binary classifier returning value 0<=p<=1).

    Usage:

    classifier=AhaabAtomClassifierBinder(feature_size)

    > Produces a neural network with input feature size defined by the input
      variable "feature size".
    '''
    def __init__(self,feature_size,hidden_layer1=512,hidden_layer2=256):
=======
    from non-binders (binary classifier returning value 0p<=1)
    '''
    def __init__(self,feature_size=140,hidden_layer1=256,hidden_layer2=128):
>>>>>>> 4aaaf06f474a91f00483a2a78237897414a871db
        super().__init__()
        self.hidden1=nn.Linear(feature_size,hidden_layer1)
        self.act1=nn.ReLU()
        self.hidden2=nn.Linear(hidden_layer1,hidden_layer2)
        self.act2=nn.ReLU()
        self.output=nn.Linear(hidden_layer2,1)
        self.act3=nn.Sigmoid()
    
    def forward(self,x):
        x=self.act1(self.hidden1(x))
        x=self.act2(self.hidden2(x))
        x=self.act3(self.output(x))
        return x