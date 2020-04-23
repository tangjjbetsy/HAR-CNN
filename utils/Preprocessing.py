import pandas as pd
import numpy as np
import torch
from utils.constants import *
from torch.utils.data import Dataset

class Preprocessing():
    """ A Preprocessing Method Collection """
    def __init__(self):
        self.fd = {"train":TRAINFOLD, "test":TESTFOLD}
        self.fl = {"train":TRAINLIST, "test":TESTLIST}
        self.ft = {"train":TRAINFEATURE, "test":TESTFEATURE}
    

    def _load_data(self, file_path, data_type='original'):
        if data_type == 'original':
            data = pd.read_csv(file_path, sep="\s+").iloc[:,0:WINDOW_WIDTH]
        elif data_type == 'features':
            data = pd.read_csv(file_path, sep="\s+")
        return data

    def _choose_status(self, status="train"):
        if status == "train":
            return self.fd["train"], self.fl["train"], self.ft["train"]
        else:
            return self.fd["test"], self.fl["test"], self.ft["test"]
    
    def _get_one_hot(self, label, N):
        size = list(label.size())
        label = label.view(-1) - 1
        ones = torch.sparse.torch.eye(N)
        ones = ones.index_select(0, label)   
        size.append(N) 
        return ones.view(*size)
    
    def _concatenate(self, pair, dataX, dataY):
        actionA = dataX[np.argwhere(dataY==pair[0])[:,0]]
        actionB = dataX[np.argwhere(dataY==pair[1])[:,0]]
        for i in range(len(actionA)):
            cut = np.random.randint(0, 128)
            selection = np.random.randint(0,len(actionB))
            actionA[i,:,cut:]= actionB[selection,:,cut:]
        return len(actionA), actionA
    
    def _stack(self, dataX, fl, fd):
        for i in range(1, len(fl)):
            fp = fd + fl[i]
            new = self._load_data(fp)
            if i == 1:
                dataX = np.stack([dataX,new],axis=len(dataX.shape))
            else:
                dataX = np.dstack([dataX,new])
        dataX = np.swapaxes(dataX,1,2)
        return dataX
    
    def original(self, status):
        fd, fl, ft = self._choose_status(status)
        dataX = self._load_data(fd + fl[0])
        dataX = self._stack(dataX, fl, fd)
        dataX = torch.FloatTensor(dataX)
        dataY = torch.LongTensor(np.asarray(self._load_data(ft[1])))
        # dataY = self._get_one_hot(dataY, 6)
        return dataX, dataY
    
    def features(self, status):
        fd, fl, ft = self._choose_status(status)
        dataX = torch.FloatTensor(np.asarray(self._load_data(ft[0], 'features').iloc[:,0:NUM_FEATURES_USED]))
        dataY = torch.LongTensor(np.asarray(self._load_data(ft[1], 'features')))
        return dataX, dataY

    def statistics(self, status):
        fd, fl, ft = self._choose_status(status)
        dataX = np.asarray(self._load_data(ft[0], 'features').iloc[:,0:NUM_FEATURES_USED])
        dataY = np.asarray(self._load_data(ft[1], 'features'))
        return dataX, dataY

    def trans(self, status):
        fd, fl, ft = self._choose_status(status)
        dataX = self._load_data(fd + fl[0])
        dataX = self._stack(dataX, fl, fd)
        dataY = np.asarray(self._load_data(ft[1]).values)
        for i in range(NUM_CLASSES):
            pair = PAIR[i+1]
            num, concatenation = self._concatenate(pair, dataX, dataY)
            if i == 0:
                outputX = torch.FloatTensor(concatenation)
                outputY = torch.LongTensor(np.repeat(i+1, num))
            else:
                outputX = torch.cat([outputX, torch.FloatTensor(concatenation)])
                outputY = torch.cat([outputY, torch.LongTensor(np.repeat(i+1, num))])
        return outputX, outputY


if __name__ == "__main__":
    prepare_data = Preprocessing()
    training_data_X, training_data_y = prepare_data.original("train")
    testing_data_X, testing_data_y = prepare_data.original("test")
    print(training_data_X.shape, training_data_y.shape)
    print(testing_data_X.shape, training_data_y.shape)

