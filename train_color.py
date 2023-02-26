import os

import get_ref
from helper.parser import train_parser

import torch
import torch.optim.lr_scheduler as lr_scheduler

from tensorboardX import SummaryWriter
from torchvision.models import resnet34, resnet101
from torchvision.models.resnet import BasicBlock, Bottleneck

from utils import helper, loss
from utils.color_space_convert import lab_to_rgb
from models.networks import AttentionExtractModule

def train(opt):
  ''' TODOs: 
    -LOAD DATASET
    -CREATE ATTENUATION MODEL
    -ADD OPTIMIZER
    -ADD SCHEDULER
    -ADD LOADING MODEL MECHANIC

  '''
  
  base_resnet = resnet34(pretrained=True)
  att_model = AttentionExtractModule(BasicBlock, [3, 4, 6, 3])
  att_model.load_state_dict(base_resnet.state_dict())
  att_model.eval()

  model = helper.create_model()

  if opt.use_gpu:
    att_model.cuda()
    #model.cuda()
  
  #criterion = helper.setup_loss(hypes)
  #optimizer = helper.setup_optimizer(hypes['train_params']['solver'], model)
  #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


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
  train(opt)