import os

#For modeling
import tensorflow as tf
from tensorflow import keras
#from keras.callbacks import TensorBoard
#import pandas as pd 

#For vector/array operations
import numpy as np
from numpy import asarray
from random import sample
import random 
from random import shuffle
import math

import datetime

import tensorboard
print('Tensorboard', tensorboard.__version__)
print('Tensorflow version', tf.__version__)


data_dir = os.path.join(os.getcwd(), os.pardir, "data")     #your job: define data_dir
work_dir = os.path.join(os.getcwd(), "work_data")  # directory: combination of current directory and saved model string


#Load saved numpy arrays instead of computed ones

train_images=np.load(os.path.join(work_dir, "train_images.npy"), mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
train_labels=np.load(os.path.join(work_dir, "train_labels.npy"), mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
val_images=np.load(os.path.join(work_dir, "val_images.npy"), mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
val_labels=np.load(os.path.join(work_dir, "val_labels.npy"), mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
test_images=np.load(os.path.join(work_dir, "test_images.npy"), mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
test_labels=np.load(os.path.join(work_dir, "test_labels.npy"), mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

#3.4 Summarize training, validation, and test data.
#Notice the datasets are not balanced (e.g. proportion in supercategory <50%).
#This is especially important for what evaluation metrics to use.
print('Training Number of instances: {:d}, Proportion in supercategory: {:f}%'.format(len(train_labels), 100*sum(train_labels)/len(train_labels)))
print('Validation Number of instances: {:d}, Proportion in supercategory: {:f}%'.format(len(val_labels), 100*sum(val_labels)/len(val_labels)))
print('Test Number of instances: {:d}, Proportion in supercategory: {:f}%'.format(len(test_labels), 100*sum(test_labels)/len(test_labels)))

#3.5 We scale these values to a range of 0 to 1 before feeding to the neural network model !  
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

#4. Model Training

#4.2 Configure model using built-in keras functionality
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1000, 1000)),
    keras.layers.Dense(128, activation=tf.nn.relu),
#    keras.layers.Dense(128, activation=tf.nn.relu), # one can add additional layers
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

#4.4 Configure loss function using built-in keras functionality
optimizer = keras.optimizers.Adam(lr=0.001) #learning rate is a hyper-parameter that we are fixing for now
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


#Print a summary of the model to understand model complexity
print(model.summary())

#4.5 Train model based on previously defined parameters

model.fit(train_images,
          train_labels,
          epochs=32, #Number of epochs is a hyper-parameter we are fixing for now
          callbacks=[tensorboard_callback]) #Default batch size is 32 (when not specified)

#Define directory for storing models
model_dir = os.path.join(os.getcwd(), "saved_model")  # directory: combination of current directory and saved model string
mod_file = os.path.join(model_dir, "my_model")        # path to the trained model to save

#Create directory if it doesn't already exist
try:
    os.stat(model_dir)
except:
    os.mkdir(model_dir)
print(mod_file)

#Save trained model
model.save(mod_file)

