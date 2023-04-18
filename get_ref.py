import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import cv2
import os
import tensorflow as tf
import tensorflow_hub as hub

from keras.layers import Flatten, Dense, Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
from scipy import spatial
from PIL import Image

from utils.parser import refdata_parser

"""
# Load pre-trained VGG16 model
VGG16 = tf.keras.applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))

# Extract vector from layer "fc2"
b_model = Model(VGG16.input, outputs = VGG16.get_layer('fc2').output)
#b_model.summary()"""

model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

image_size = 224
IMAGE_SHAPE = (image_size, image_size)

layer = hub.KerasLayer(model_url, input_shape = IMAGE_SHAPE+(3,))
model = tf.keras.Sequential([layer])

# Get feature vector of an image
def get_feature_vector(_img):
  img = cv2.resize(_img, (image_size, image_size))
  #feature_v = b_model.predict(img.reshape(1, image_size, image_size, 3))
  feature_v = model.predict(_img[np.newaxis, ...])
  feature_v = np.array(feature_v)
  feature_v = feature_v.flatten()
  
  return feature_v

# Calculate euclidean similarity
def calculate_similarity(v1, v2):
  return np.linalg.norm(v1 - v2)

# Get similarity between two vectors
def get_image_similarity(img1, img2):
  feature_v1 = get_feature_vector(img1)
  feature_v2 = get_feature_vector(img2)

  return calculate_similarity(feature_v1, feature_v2)

# Get top similar image for each image in a directory
def find_all_top_images(img_dir_path, output_path):
  img_dir = os.listdir(img_dir_path)
  img_to_ref = []

  counter = 0
  for img_file_name in img_dir:
    if (counter + 1) % 100 == 0:
      print(counter + 1)
    top_img_path = find_top_image(img_file_paths[counter], img_feature_vect=img_vects[counter],verbose=False)[0]
    #print(img_file_paths[counter] + " for " + top_img_path)
    img_to_ref.append([img_file_paths[counter], top_img_path])
    counter += 1

  print("Writing map file to " + output_path)
  with open(output_path, 'w') as f:
      for pair in img_to_ref:
        f.write(pair[0] + "," + pair[1] +"\n")

  return img_to_ref
  

# Get top 2 similar images from a dataset folder
def find_top_image(img_path, img_feature_vect=False, verbose=True):
  if verbose:
    print("Img path = " + str(img_path))
  
  if type(img_feature_vect) == bool:
    img = Image.open(img_path).convert('L').resize(IMAGE_SHAPE)  #1
    img = np.stack((img,)*3, axis=-1)                       #2
    img = np.array(img)/255.0

    # Feature vector of image to be compared with dataset
    img_vect = get_feature_vector(img)
  else:
    img_vect = img_feature_vect
  
  top_score = 9999999999
  top2_score = 9999999999
  
  top_image_path = ""
  top2_image_path = ""
  counter = 0

  for img_ref_vect in img_ref_vects:
    score = calculate_similarity(img_vect, img_ref_vect)
    if score < 0.01: # Skip same image
      counter += 1
      continue
    if score < top_score:
      top2_image_path = top_image_path
      top2_score = top_score
      top_image_path = img_ref_file_paths[counter]
      top_score = score
    elif score < top2_score and img_ref_file_paths[counter] != top_image_path:
      top2_image_path = img_ref_file_paths[counter]
      top2_score = score
    counter += 1
  
  if verbose == True:
    if top_score > 14:
      print("Could not find a good match")
    print("Top1 Image = " + top_image_path)
    print("Top2 Image = " + top2_image_path)
    print("Top Score = " + str(top_score))
    print("Top2 Score = " + str(top2_score))
  
  return [top_image_path, top2_image_path]

def feature_reader(ref_feature_dir_path, ref_dir, feature_dir_path=False, img_dir=False):
  ref_feature_dir = os.listdir(ref_feature_dir_path)
  print("Ref Feature File count = " + str(len(ref_feature_dir) - 1))
  
  global img_ref_vects
  global img_ref_file_paths
  global img_vects
  global img_file_paths
  img_ref_file_paths = []
  img_ref_vects = np.empty([len(ref_feature_dir) - 1, 1280,])
  img_file_paths = []
  img_vects = np.empty([len(ref_feature_dir) - 1, 1280,])

  counter = 0
  print("Reading features")
  for file_name in ref_feature_dir:
    if (counter + 1) % 100 == 0:
      print(counter + 1)
    # Gets image file name without .txt in the end
    file_path = os.path.join(ref_feature_dir_path, file_name)
    if file_path[-3:] != "txt":
      if file_path[-6:] == "ignore":
        print("Ignoring .gitignore")
        continue
      raise Exception("INVALID FILE IN FEATURES FOLDER: " + file_path)
    img_ref_vects[counter] = parse_feature_file(file_path)
    ref_img_path = os.path.join(ref_dir, file_name[:-4])
    img_ref_file_paths.append(ref_img_path)
    counter += 1
    if counter == (len(ref_feature_dir) - 1):
      break
  
  if feature_dir_path == "" or feature_dir_path == False:
    return
  
  feature_dir = os.listdir(feature_dir_path)
  print("Feature File count = " + str(len(feature_dir) - 1))

  counter = 0
  print("Reading features")
  for file_name in feature_dir:
    if (counter + 1) % 100 == 0:
      print(counter + 1)
    # Gets image file name without .txt in the end
    file_path = os.path.join(feature_dir_path, file_name)
    if file_path[-3:] != "txt":
      if file_path[-6:] == "ignore":
        print("Ignoring .gitignore")
        continue
      raise Exception("INVALID FILE IN FEATURES FOLDER: " + file_path)
    img_vects[counter] = parse_feature_file(file_path)
    img_path = os.path.join(img_dir, file_name[:-4])
    img_file_paths.append(img_path)
    counter += 1

# Gets the vectory feature array from a text file
def parse_feature_file(file_path):
  with open(file_path) as f:
    lines = f.read().splitlines()
  x = np.array(lines)
  y = x.astype(np.double)
  y = y.reshape(1280,)
  return y

# Get feature vectors for all images in an image dir
def get_feature_vectors(img_dir_path, output_path):
  print("Processing feature vectors from: " + img_dir_path)
  img_dir = os.listdir(img_dir_path)
  counter = 0

  for file_name in img_dir:
    file_path = os.path.join(img_dir_path, file_name)
    #img = cv2.imread(file_path)
    img = Image.open(file_path).convert('L').resize(IMAGE_SHAPE)  #1
    img = np.stack((img,)*3, axis=-1)                       #2
    img = np.array(img)/255.0                               #3

    counter += 1
    print(counter)

    try:
      dummy = img.shape
    except:
      print("Invalid image file while getting similarity: " + file_name)

    img_vect = get_feature_vector(img)

    with open(output_path + "/" + file_name + '.txt', 'w') as f:
      for element in img_vect:
        f.write(str(element) + '\n')

def run_operation(opt):
  opt = refdata_parser()
  print(opt.ref_feature_dir)
  print(opt.data_dir)
  if opt.op == "get_features":
    print("Getting Features")
    get_feature_vectors(opt.data_dir, opt.feature_dir)
    print("Done")
  elif opt.op == "get_ref":
    feature_reader(opt.ref_feature_dir, opt.ref_dir, False, False)
    print("Finding top 2 images")
    return find_top_image(opt.img_path, verbose=True)
  elif opt.op == "get_refs":
    feature_reader(opt.ref_feature_dir, opt.ref_dir, opt.feature_dir, opt.img_path)
    print("Finding top images")
    return find_all_top_images(opt.img_path, opt.write_to)

if __name__ == '__main__':
  # load training configuration from yaml file
  opt = refdata_parser()
  run_operation(opt)


    

    
