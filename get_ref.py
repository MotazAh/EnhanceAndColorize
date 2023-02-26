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

# Calculate cosine similarity
def calculate_similarity(v1, v2):
  return 1 - spatial.distance.cosine(v1, v2)

# Get similarity between two vectors
def get_image_similarity(img1, img2):
  feature_v1 = get_feature_vector(img1)
  feature_v2 = get_feature_vector(img2)

  return calculate_similarity(feature_v1, feature_v2)

# Get top 2 similar images from a dataset folder
def find_top_images(img, feature_dir_path):
  top_score = -1
  top2_score = -1
  counter = 0

  feature_dir = os.listdir(feature_dir_path)
  print("File count = " + str(len(feature_dir)))

  # Feature vector of image to be compared with dataset
  img_vect = get_feature_vector(img)

  for file_name in feature_dir:
    counter += 1
    if (counter % 100) == 0:
      print(counter)

    # Gets image file name without .txt in the end
    file_path = os.path.join(feature_dir_path, file_name)
    if file_path[-3:] != "txt":
      continue
    img_ref_vect = parse_feature_file(file_path)

    score = calculate_similarity(img_vect, img_ref_vect)

    if score > top_score:
      top_image_name = file_name[:-4]
      top_score = score
    elif score > top2_score:
      top2_image_name = file_name[:-4]
      top2_score = score

  top_image_paths = [top_image_name, top2_image_name]
  return top_image_paths

# Gets the vectory feature array from a text file
def parse_feature_file(file_path):
  with open(file_path) as f:
    lines = f.read().splitlines()
  x = np.array(lines)
  y = x.astype(np.double)
  y = y.reshape(1, 4096)
  return y

# Get feature vectors for all images in an image dir
def get_feature_vectors(img_dir_path):
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

    with open('Dataset/feature_vectors/' + file_name + '.txt', 'w') as f:
      for element in img_vect:
        f.write(str(element) + '\n')

"""
if __name__ == '__main__':
    print("test")
    # load training configuration from yaml file
    opt = parser.data_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    # gpu setup
    use_gpu = hypes['train_params']['use_gpu']
    if use_gpu:
        # TODO: Multi-Gpu implementation
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hypes['train_params']['gpu_id'])

    if opt.crack_net:
        # todo: add restoration training later
        pass

    elif 'gan' not in hypes:
        train_nogan.train(opt, hypes, use_gpu)

    else:
        print("Starting")
        train_nogan.train(opt, hypes, use_gpu)
        # todo: add gan training later
        pass
        # train_gan.train(opt, hypes, use_gpu)"""
