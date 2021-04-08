import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from tensorflow.keras.layers import Input, Dense, Dropout #There are the two layer types we need
from tensorflow.keras.models import Model        #This contains functions and structures for building a model

# Read in the data from the .mat files
art = scipy.io.loadmat('fingerlines_art.mat')
artFree = scipy.io.loadmat('fingerlines_artFree.mat')

# Create variables to hold the train, validate, and test portions of the data
artTrain = []
artFreeTrain = []
artTest = [];
artFreeTest = [];

n = len(art['art'])
m = len(artFree['artFree'])



# Split data for train, validate, and test
artTrain = np.array(art['art'][0:int(n*0.9)][:])
artTest = np.array(art['art'][int(n*0.9):][:])

artFreeTrain = np.array(artFree['artFree'][0:int(m*0.9)][:])
artFreeTest = np.array(artFree['artFree'][int(m*0.9):][:])

# Normalize the data
artTrain = preprocessing.normalize(artTrain)
artTest = preprocessing.normalize(artTest)
artFreeTrain = preprocessing.normalize(artFreeTrain)
artFreeTest = preprocessing.normalize(artFreeTest)





model = tf.keras.Sequential()
layer_1 = Dense(1)
model.add(layer_1)

model.compile(optimizer='adam', loss='mean_squared_error')

for x in range(len(artTrain)):
	x_test = np.array(artTrain[x])
	y_test = np.array(artFreeTrain[x])
	model.fit(x_test,y_test, epochs=1)


results = []
for x in range(len(artTest)):
	results.append(model.predict(artTest[x]))



plt.subplot(3,1,1)
plt.plot(results[0], label='predicted')
plt.title('predicted')
plt.subplot(3,1,2)
plt.plot(artTest[0], label='data with noise')
plt.title('data with noise')
plt.subplot(3,1,3)
plt.plot(artFreeTest[0], label='actual data')
plt.title('actual data')

plt.show()












