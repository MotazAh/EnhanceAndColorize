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


# Load pre-trained VGG16 model
vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))

# Extract vector from layer "fc2"
b_model = Model(vgg16.input, outputs = vgg16.get_layer('fc2').output)

# Get feature vector of an image
def get_feature_vector(_img):
  img = cv2.resize(_img, (224, 224))
  feature_v = b_model.predict(img.reshape(1, 224, 224, 3))
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
def find_top_images(img, img_dir):
  top1_score = 0
  top2_score = 0
  top3_score = 0

  for file_name in os.listdir(img_dir):
    file_path = os.path.join(img_dir, file_name)
    img_ref = cv2.imread(file_path, 0)

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















