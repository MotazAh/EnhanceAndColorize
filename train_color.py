import os

import get_ref
from helper.parser import train_parser

def train():
  pass

if __name__ == '__main__':
  # load training configuration from yaml file
  opt = train_parser()
  
  # Check gpu
  opt.use_gpu
  if opt.use_gpu:
    print("Using gpu")
  else:
    print("Not using gpu")
  
  print("Starting")
  train()