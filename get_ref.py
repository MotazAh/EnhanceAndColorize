import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import cv2
import os
import tensorflow as tf

from keras.layers import Flatten, Dense, Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
from scipy import spatial


# Load pre-trained InceptionV3 model
effNetB4 = tf.keras.applications.EfficientNetB4(weights='imagenet', include_top=True, pooling='max', input_shape=(380, 380, 3))
#effNetB4 = tf.keras.applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))

# Extract vector from layer "fc2"
#for layer in effNetB4.layers:
#    print(layer.name)
b_model = Model(effNetB4.input, outputs = effNetB4.get_layer('block3a_se_squeeze').output)
b_model.summary()

# Get feature vector of an image
def get_feature_vector(_img):
  img = cv2.resize(_img, (380, 380))
  feature_v = b_model.predict(img.reshape(1, 380, 380, 3))
  return feature_v

# Calculate cosine similarity
def calculate_similarity(v1, v2):
  return 1 - spatial.distance.cosine(v1, v2)

# Get similarity between two images
def get_image_similarity(img1, img2):
  feature_v1 = get_feature_vector(img1)
  feature_v2 = get_feature_vector(img2)

  return calculate_similarity(feature_v1, feature_v2)

# Get top 3 similar images to an image from a dataset folder
def find_top_images(img, img_dir_path):
  top1_score = 0
  top2_score = 0
  top3_score = 0
  counter = 0

  img_dir = os.listdir(img_dir_path)
  print("File count = " + str(len(img_dir)))

  for file_name in img_dir:
    file_path = os.path.join(img_dir_path, file_name)
    img_ref = cv2.imread(file_path)
    
    counter += 1
    print(counter)

    try:
      dummy = img_ref.shape
    except:
      print("Invalid image file while getting similarity: " + file_name)

    score = get_image_similarity(img, img_ref)
    if score > top1_score:
      top1_image = img_ref
      top1_path = file_path
      top1_score = score
    elif score > top2_score:
      top2_image = img_ref
      top2_path = file_path
      top2_score = score
    elif score > top3_score:
      top3_image = img_ref
      top3_path = file_path
      top3_score = score
  top_img_list = [top1_image, top2_image, top3_image]
  top_path_list = [top1_path, top2_path, top3_path]

  return top_img_list, top_path_list















