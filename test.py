import pandas as pd
import numpy as np
from utils.Preprocessing import *

def load_data(file_path):
    data = pd.read_csv(file_path, sep="\s+")
    return data

# X = load_data("../data/train/X_train.txt")
# y = load_data("../data/train/y_train.txt")
# y = np.asarray(y.values)
# actionA = X.iloc[np.argwhere(y==5)[:,0]]
# print(len(actionA))

a = Preprocessing()
X, Y = a.trans('train')
print(X.shape)
X, Y = a.trans('test')
print(X.shape)
