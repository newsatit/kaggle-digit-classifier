import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd
import matplotlib.pyplot as plt

# Load the training data
train_set = pd.read_csv('./data-sets/train.csv')
# extract labels 
train_label = train_set.label
train_label = np.array(train_label)
# get training features
train_feat = np.array(train_set.iloc[:, 1:])

# get number of training set
num_train = train_feat.shape[0]

# reshape the training data
x_train = train_feat.reshape((num_train, 28, 28, 1))
x_train = x_train/225
y_train = train_label.reshape(-1, 1)


# create Keras Sequential Convolutional Neural Network
model = Sequential()
# Convolutional & ReLu and Max Pooling 
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Convolutional & ReLu and Max Pooling 
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# fully connect neurons
model.add(Flatten())
model.add(Dense(units=256, activation="relu"))
# randomly setting to zero to prevent overfittinn
model.add(Dropout(0.1))
# reduce to 10 labels
model.add(Dense(units=10, activation='softmax'))

# train the model
# model.compile(optimizer='adam', 
#               loss='sparse_categorical_crossentropy', 
#               metrics=['accuracy'])
# model.fit(x=x_train,y=y_train, epochs=1)

#Load the testing data
test_set = pd.read_csv('./data-sets/test.csv')
# get testing features
test_feat = np.array(test_set)
# get number of training set
num_test = test_feat.shape[0]
# reshape the testing data
x_test = test_feat.reshape((num_test, 28, 28, 1))
x_test = x_test/225

# predict
predictions = model.predict(np.array(x_test)).argmax(axis=1)








