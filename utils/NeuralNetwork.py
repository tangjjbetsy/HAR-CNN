import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.constants import *
import numpy as np
import pandas as pd

class Conv1DNet(nn.Module):
	
    def __init__(self, fc1_input_dim):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(9, 64, 3)
        self.conv2 = nn.Conv1d(64, 64, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64*fc1_input_dim, 100)
        self.fc2 = nn.Linear(100, NUM_CLASSES)
        self.max = nn.MaxPool1d(2)
    
    def _class_name(self):
        return "Conv1DNet"

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = self.max(x)
        x = F.dropout(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class DNN(nn.Module):
    	
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, NUM_CLASSES)
    
    def _class_name(self):
        return "DNN"


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AutoEncoder(nn.Module):

    def __init__(self, input_dim, hidden_size, mode):
        super(AutoEncoder, self).__init__()
        if mode == 1:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(True))
            
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, input_dim),
                nn.ReLU(True))
        elif mode ==2:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, int(input_dim/2)),
                nn.ReLU(True),
                nn.Linear(int(input_dim/2), hidden_size),
                nn.ReLU(True))
            
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size,int(input_dim/2)),
                nn.ReLU(True),
                nn.Linear(int(input_dim/2), input_dim),
                nn.ReLU(True))

        self.sigmoid = nn.LogSigmoid()
        
        self.hidden_size = hidden_size
        self.input_dim = input_dim

    def _class_name(self):
        return "AutoEncoder"+str(self.hidden_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.sigmoid(x)
        x = self.decoder(x)
        return x

class StackedAutoEncoder(nn.Module):
    def __init__(self, AE1, AE2):
        super(StackedAutoEncoder, self).__init__()
        self.layer1 = nn.Sequential(*list(AE1.children())[:-2])
        self.layer2 = nn.Sequential(*list(AE2.children())[:-2])
        self.layer3 = nn.Linear(AE2_DIM, NUM_CLASSES)
    
    def _class_name(self):
        return 'StackedAutoEncoder'
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


if __name__ == "__main__":
    net = Conv1DNet()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    

    
