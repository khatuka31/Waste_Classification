#Import libraries and modules 

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)    # ignores warnings about future version of numpy

#For JSON data
import json

#For interacting with operating system 
import os

#For copying files
import shutil

#For vector/array operations
import numpy as np
from numpy import asarray
from random import sample
import random 
from random import shuffle
import math
from time import time
import datetime

#For loading and preprocessing images
from PIL import Image 
import matplotlib.pyplot as plt

print('Numpy version', np.__version__)
start_time = time()

#1. Dataset exploration 
#Current directory 
print(os.getcwd())

#1.1 Load annotations: annotations.json[1] contains annotations in COCO[2] format

data_dir = os.path.join(os.getcwd(), os.pardir, "data")     #your job: define data_dir
anno_file = os.path.join(data_dir, "annotations.json")      #your job: path for *.json file

with open(anno_file, "r") as f: # annotations.json is a nested dictionary (keys are mapped to another dictionary within original dictionary)                     
    annotations = json.load(f)  # your job: parse JSON string 

#1.2 Explore annotations dictionary
#Exercise: explore each annotation key

#Number of images
len(annotations["images"])

#Number of annotations 
len(annotations["annotations"]) 

#1.2.1 Question to audience: Explain why there are more annotations than images
#Image information
annotations["images"][320]

annotations["images"][1210]  #your job: explore the annotation

#1.2.2 Exercise: explore annotation for image_id 6
for anno in annotations ["annotations"]:
    if anno["image_id"]==6:
        print(anno)

annotations["categories"][11]

for anno in annotations ["scene_annotations"]:
    if anno["image_id"]==6:
        print(anno)

#1.2.3 Exercise for advanced participants: explore nested dictionary in details.


#2. Preprocessing
# For simplicity, we create a simplified dictionary w.r.t to each image and its associated categories. We store only a subset of information, e.g. 'image_id', 'file_name', 'height', 'width', 'category_ids', 'category_names', and 'super_categories'

#2.1 Open annotation file and read into memory
with open(anno_file, "r") as f:
    annotations = json.load(f)

# 2.2 Prepare category id to name mappings. Items are ordered by category_id, so you can get the
# category name of a category_id via the category_id, e.g.
# via annotations["categories"][category_id]
categories = annotations["categories"]

#2.3 Create new python dictionary with subset of relevant information (e.g. image -> category data)
data = {}
for i, item in enumerate(annotations["annotations"]):
    #Map image_id to image filename using the "images" part of the dataset.
    image_id = item["image_id"]
    image_info = annotations["images"][image_id]
    file_name = image_info["file_name"]
    height,width = image_info["height"], image_info["width"]
    
    #Map category_id of instance to category name
    category_id = item["category_id"]
    category_info = categories[category_id]
    category_name = category_info["name"]
    super_category = category_info["supercategory"]
    
    #A labeled image can have multiple categories, so check if we have already added to the dictionary (e.g. if it's in the keys)
    if image_id in data.keys():
        data[image_id]["category_ids"].add(category_id)
        data[image_id]["category_names"].add(category_name)
        data[image_id]["super_categories"].add(super_category)
    else:
        data[image_id] = {"file_name": file_name, "category_ids": {category_id}, "image_id": image_id, "height": height, "width": width, "category_names": {category_name}, "super_categories": {super_category}}

len(data)
data[320] #labels in particular instance in our dataset (image_id = 320)

#3. Binary Classification
#We can construct a binary classification problem in a one vs all setting, e.g. does this image contain a specific 
# supercategory or not. Let's create the numpy arrays corresponding to the images and labels that we can use for training.

#3.1 Split data into training, validation, and test

data_ids = list(data.keys())


#Configure proportion of training, validation, and test data
train_perc = 0.8
val_perc = 0.1
test_perc = 0.1
train_size=int(len(data_ids)*train_perc)
val_size=int(len(data_ids)*val_perc)
train_ids, val_ids, test_ids = (
    data_ids[:train_size],
    data_ids[train_size : train_size + val_size],
    data_ids[train_size + val_size :],
    )

print("Number of images in training dataset:", len(train_ids))
print("Training image_ids:", train_ids)

print("Number of images in validation dataset:", len(val_ids))
print("Validation image_ids:", val_ids)

print("Number of images in  dataset:", len(test_ids))
print("test image_ids:", test_ids)

print(len(test_ids))

#3.2 Define helper function for loading data and converting to numpy arrrays

def load_data(ids, data, supercategory):
    num_instances = len(ids)
    max_height, max_width =  1000, 1000 #TBD revisit max heigh and width
    labels = np.zeros((num_instances,))
    images = np.zeros((num_instances,max_height,max_width))

    for i, image_id in enumerate(ids):
        #Convert labels into a binary classification problem (e.g. 0 or 1 depending on the super_category)
        if supercategory in data[image_id]["super_categories"]:
            labels[i] = 1

        #Load images into numpy arrays
        try: 
            image = Image.open(os.path.join(data_dir,data[image_id]["file_name"])).convert("L")  # Grayscale
            image = image.resize(size=(max_height, max_width))
            images[i] = asarray(image) #Convert and store as numpy array

        except Exception as e:
            print(e) #Use this to catch and print exceptions
    return images, labels

#3.3 Define supercategory of interest (in this case 'Bottle') and load training, validation, and test data 
supercategory = "Bottle"
train_images, train_labels = load_data(train_ids, data, supercategory)
val_images, val_labels = load_data(val_ids, data, supercategory)
test_images, test_labels = load_data(test_ids, data, supercategory)

work_dir = os.path.join(os.getcwd(), "work_data")  # directory: combination of current directory and saved model string
arr_file_trimg = os.path.join(work_dir, "train_images")        # path to the trained model to save  
arr_file_bl = os.path.join(work_dir, "train_labels")        # path to the trained model to save
arr_file_valimg = os.path.join(work_dir, "val_images")        # path to the trained model to save  
arr_file_vallabel = os.path.join(work_dir, "val_labels")        # path to the trained model to save
arr_file_testimg = os.path.join(work_dir, "test_images")        # path to the trained model to save  
arr_file_testlabel = os.path.join(work_dir, "test_labels")        # path to the trained model to save

#Check if directory exists. If not, create it
try:
    os.stat(work_dir)
except:
    os.mkdir(work_dir)

np.save(arr_file_trimg, train_images, allow_pickle=False, fix_imports=False)
np.save(arr_file_bl, train_labels, allow_pickle=False, fix_imports=False)
np.save(arr_file_valimg, val_images, allow_pickle=False, fix_imports=False)
np.save(arr_file_vallabel, val_labels, allow_pickle=False, fix_imports=False)
np.save(arr_file_testimg, test_images, allow_pickle=False, fix_imports=False)
np.save(arr_file_testlabel, test_labels, allow_pickle=False, fix_imports=False)

end_time = time()
total_time = end_time - start_time
print("Execution time:", str(datetime.timedelta(seconds=total_time)))

