import numpy as np
import keras
from keras.models import Sequential
import pandas as pd

# Load the training data
train_set = pd.read_csv('./data-sets/train.csv')
# extract and reshape labels from column vector to 2D numpy.ndarray
train_label = train_set.label
train_label = np.array(train_label).shape(-1, 1)
# get normalized training features
train_feat = np.array(train_set.iloc[:, 1:])/255


