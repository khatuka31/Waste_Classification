#(Copyright: Khatuna Kakhiani for HLRS)

# Ignore warnings
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)    # ignores warnings about future version of numpy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#For interacting with operating system 
import os

#For modeling 
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

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

data_dir = os.path.join(os.getcwd(), 'data_dir')     #define data_dir
work_dir = os.path.join(os.getcwd(), "work_data")  # directory: combination of current directory and saved model string


#Exercise 3.4: Load previously stored numpy arrays for training, validation, and test images and labels


#3.4 Summarize training, validation, and test data.
#Notice the datasets are not balanced (e.g. proportion in supercategory <50%).
#This is especially important for what evaluation metrics to use.
print('Training Number of instances: {:d}, Proportion in supercategory: {:f}%'.format(len(train_labels), 100*sum(train_labels)/len(train_labels)))
print('Validation Number of instances: {:d}, Proportion in supercategory: {:f}%'.format(len(val_labels), 100*sum(val_labels)/len(val_labels)))
print('Test Number of instances: {:d}, Proportion in supercategory: {:f}%'.format(len(test_labels), 100*sum(test_labels)/len(test_labels)))

# Exercise 3.5: Normalize pixel values in train, validation, and test images to a scale of 0 to 1 

#4. Model Training

# Exercise 4.2: Configure model output layer and activation function 
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1000, 1000)), 
    keras.layers.Dense(128, activation=tf.nn.relu),
#    keras.layers.Dense(128, activation=tf.nn.relu), # one can add additional layers
    #TODO for exercise)
])

#4.4 Configure loss function using built-in keras functionality
optimizer = keras.optimizers.Adam(lr=0.001) #learning rate is a hyper-parameter that we are fixing for now
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


# Exercise 4.4: Verify model was configured correctly by using summary functionality

# 4.5: train model based on previously defined parameters, use epochs < 3
# Exercise 4.5: Set hyper parameters of batch size and number of epochs for training in fit() method

model.fit(train_images,
          train_labels,
          #TODO exercise
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
