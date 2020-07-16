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

#5. Model Evaluation

#Define directory for storing models
model_dir = os.path.join(os.getcwd(), "saved_model")  # directory: combination of current directory and saved model string
mod_file = os.path.join(model_dir, "my_model")        # path to the trained model to save

#5.1 Load model
model = tf.keras.models.load_model(mod_file)

#5.2 Determine how much performance varies between training and unseen (e.g. validation or test) data.
#Notice the large difference between training and unseen data, indicating the need for regularization.
model.evaluate(train_images, train_labels)
model.evaluate(val_images, val_labels)

#5.3 Determine appropriate evaluation metrics
#In section 3.4, we saw the dataset is imbalanced. This means accuracy is not an approriate evaluation measure.
#Let's use precision, recall, and f1 to measure the model's performance on the supercategory of interest.
# Precision = TP / TP + FP ... i.e. how many selected items are relevant
# Recall = TP / TP + FN ... i.e. how many relevant items are selected

#5.3.1 Get predicted model scores
#A prediction is an array of 2 numbers. These describe the "confidence" of the model that the image corresponds
#to bottle or no bottle (i.e each different litter).
#We can see which label has the highest confidence value using maxium value. Alternative approaches involve 
#using thresholds for predictions (depending on whether precision or recall is preferred).

prediction_scores = model.predict(val_images)

prediction_scores[80]

#5.3.2 Convert predicted scores to class labels
# argmax returns the index of the maximum value in a list
# e.g. a = [1,5,10] ... a.argmax() will return the index 2, as 10 is the highest value at index position 2
# axis determines the axis in the a matrix or tensor
# e.g. a = [[1,5,10], [11,5,1]] ... a.argmax() = 3 because argmax flattens the matrix into a 1-D vector if 
# no axis is provided. a.argmax(axis=1) will result in a list of indices for each sublist like so [2,0] as
# 10 with index 2 has the highest value in the first list, and 11 with index 0 is highest in the other list
predictions = prediction_scores.argmax(axis=1)
predictions[80]

#5.3.3 Generate detailed report with desired evaluation metrics
#Notice the lower performance on the minority class (supercategory) and how this was not captured in the accuracy metric.
from sklearn.metrics import classification_report
print(classification_report(val_labels, predictions))



