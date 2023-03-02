import os

from utils.parser import test_parser

import numpy as np
import cv2
import torch
import torch.optim.lr_scheduler as lr_scheduler
from utils.customized_transform import *

from tensorboardX import SummaryWriter
from torchvision.models import resnet34, resnet101
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils import helper, loss
from utils.color_space_convert import lab_to_rgb
from models.networks import AttentionExtractModule
from hypes_yaml import yaml_utils

from utils.color_space_convert import lab_to_rgb

def test(opt, hypes):
  
  print('loading dataset')
  # Real test is for testing real grayscale old images (Not available currently)
  transform_operation = transforms.Compose([Crop(), TolABTensor()])

  test_dataset = DatasetMaker(opt.img_path, opt.ref_path,
                                  transform=transform_operation)

  loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

  print("Setting resnet34 and attention model")
  base_resnet = resnet34(pretrained=True)
  att_model = AttentionExtractModule(BasicBlock, [3, 4, 6, 3])
  att_model.load_state_dict(base_resnet.state_dict())
  att_model.eval()

  print("Creating model")
  model = helper.create_model(hypes)

  use_gpu = opt.use_gpu
  if use_gpu:
    att_model.cuda()
    model.cuda()

   # load saved model for continue training or train from scratch
  print("Save path: " + opt.model_dir)
  if opt.model_dir:
    print("Loading previous model")
    saved_path = opt.model_dir
    init_epoch, model = helper.load_saved_model(saved_path, model, use_gpu)
  
  print('Testing Start')

  for i, batch_data in enumerate(loader_test):
    # clean up grad first
    input_batch, input_l, gt_ab, gt_l, ref_gray, ref_ab = batch_data['input_image'], \
                                                            batch_data['input_L'], \
                                                            batch_data['gt_ab'], batch_data['gt_L'], \
                                                            batch_data['ref_gray'], batch_data['ref_ab']
    # model inference and loss cal
    out_dict = model(input_l, input_batch, ref_ab, ref_gray, att_model)
    if use_gpu:
      input_batch = input_batch.cuda()
      input_l = input_l.cuda()
      gt_ab = gt_ab.cuda()
      gt_l = gt_l.cuda()
      ref_gray = ref_gray.cuda()
      ref_ab = ref_ab.cuda()
    
      output = torch.clamp(out_dict['output'], -1, 1.)
      output = lab_to_rgb(input_l, output).cuda()
      target_val = lab_to_rgb(gt_l, gt_ab).cuda()

      output_np = output.cpu().numpy()
      target_np = target_val.cpu().numpy()
    else:
      input_batch = input_batch
      input_l = input_l
      gt_ab = gt_ab
      gt_l = gt_l
      ref_gray = ref_gray
      ref_ab = ref_ab
    
      output = torch.clamp(out_dict['output'], -1, 1.)
      output = lab_to_rgb(input_l, output)
      target_val = lab_to_rgb(gt_l, gt_ab)

      output_np = output.numpy()
      target_np = target_val.numpy()

    output_np = output_np[0] * 255
    output_np = output_np.transpose(1, 2, 0)

    target_np = target_np[0] * 255
    target_np = target_np.transpose(1, 2, 0)
    print("Writing output and target images")
    cv2.imwrite("Dataset/output.jpg", cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite("Dataset/target.jpg", cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR))

class DatasetMaker(Dataset):
    """
    Dataset should have a pair of data
    """

    def __init__(self, img_path, ref_path, transform=transforms.Compose([transforms.ToTensor()])):
        """
        Args:
            :param root_dir: the path that contain all groundtruth and input images
            :param transform: callable function to do transform on origin data pair
            :param ref_json: whether load reference image from json
        """
        self.gt_images = []
        self.ref_images = []
        
        self.gt_images.append(img_path)
        self.ref_images.append(ref_path)
        """
        for folder in self.root_dir:
            gt_images = sorted([os.path.join(folder, x)
                                for x in os.listdir(folder) if x.endswith('.jpg') or x.endswith('.png')])
            print(gt_images)
            raise Exception("Test")
            if ref_json:
                ref_json_files = [os.path.join(folder, 'matches', os.path.split(x)[1][:-3] + 'json')
                                  for x in gt_images]
                self.ref_json_files += ref_json_files

            self.gt_images += gt_images"""

        self.transform = transform

    def __len__(self):
        return len(self.gt_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gt_image_name = self.gt_images[idx]

        gt_image = cv2.cvtColor(cv2.imread(gt_image_name), cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY), -1)

        data = {'input_image': input_image, 'gt_image': gt_image}
        ref_image = cv2.cvtColor(cv2.imread(self.ref_images[idx]), cv2.COLOR_BGR2RGB)
        data.update({"ref_image": ref_image})

        if self.transform:
            data = self.transform(data)

        data['image_name'] = gt_image_name
        return data

if __name__ == '__main__':
  # load training configuration from yaml file
  opt = test_parser()
  hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
  opt.use_gpu = hypes["train_params"]["use_gpu"]
  # Check gpu
  #opt.use_gpu
  if opt.use_gpu:
    print("Using gpu")
  else:
    print("Not using gpu")
  
  print("Starting")
  test(opt, hypes)