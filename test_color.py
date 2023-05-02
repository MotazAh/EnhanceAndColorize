import os

from utils.parser import test_parser, test_blank_parser

import numpy as np
import cv2
import torch
import torch.optim.lr_scheduler as lr_scheduler
from utils.customized_transform import *

from skimage.metrics import structural_similarity as SSIM

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
  transform_operation = transforms.Compose([Resize(), TolABTensor()])

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
    input_batch, input_l, gt_ab, gt_l, ref_gray, ref_ab, ref_l, input_gray = batch_data['input_image'], \
                                                                   batch_data['input_L'], \
                                                                   batch_data['gt_ab'], batch_data['gt_L'], \
                                                                   batch_data['ref_gray'], batch_data['ref_ab'], \
                                                                   batch_data['ref_l'], batch_data['input_gray']
    
    if use_gpu:
                input_batch = input_batch.cuda()
                input_l = input_l.cuda()
                gt_ab = gt_ab.cuda()
                gt_l = gt_l.cuda()
                ref_gray = ref_gray.cuda()
                ref_ab = ref_ab.cuda()
                ref_l = ref_l.cuda()
    # model inference and loss cal
    out_dict = model(input_l, input_batch, ref_ab, ref_gray, att_model)
    out_test = torch.clamp(out_dict['output'], -1., 1.)
    
    L_tensor = torch.zeros(1, 1, 256, 256)

    if use_gpu:
      L_tensor = L_tensor.cuda()
      out_ab_rgb = lab_to_rgb(L_tensor, out_test).cuda()
      out_test = lab_to_rgb(input_l, out_test).cuda()
      ref_train = lab_to_rgb(ref_l, ref_ab).cuda()
      ref_ab_rgb = lab_to_rgb(L_tensor, ref_ab).cuda()
      

      output_np = out_test[0].cpu().numpy()
      ref_np = ref_train[0].cpu().numpy()
      out_ab_np = out_ab_rgb[0].cpu().numpy()
      ref_ab_np = ref_ab_rgb[0].cpu().numpy()
    else:
      out_ab_rgb = lab_to_rgb(L_tensor, out_test)
      out_test = lab_to_rgb(input_l, out_test)
      ref_train = lab_to_rgb(ref_l, ref_ab)
      ref_ab_rgb = lab_to_rgb(L_tensor, ref_ab)
      

      output_np = out_test[0].numpy()
      ref_np = ref_train[0].numpy()
      out_ab_np = out_ab_rgb[0].cpu().numpy()
      ref_ab_np = ref_ab_rgb[0].cpu().numpy()
    
    
    
    output_np = output_np * 255
    output_np = output_np.transpose(1, 2, 0)

    ref_np = ref_np * 255
    ref_np = ref_np.transpose(1, 2, 0)

    ref_ab_np = ref_ab_np * 255
    ref_ab_np = ref_ab_np.transpose(1, 2, 0)

    out_ab_np = out_ab_np * 255
    out_ab_np = out_ab_np.transpose(1, 2, 0)

    print("Writing output, target and ref images")

    out_gray = cv2.cvtColor(output_np, cv2.COLOR_BGR2GRAY)
    gt_gray = cv2.imread(opt.img_path, 0)

    output_np_FAT = np.concatenate((input_gray[0], ref_np, output_np, ref_ab_np, out_ab_np), 1)

    # Convert images to grayscale
    
    
    print(out_gray.shape)
    print(gt_gray.shape)

    #(score, diff) = SSIM(out_gray, gt_gray, full=True)

    #print(score)

    cv2.imwrite("Dataset/output.jpg", cv2.cvtColor(output_np_FAT, cv2.COLOR_RGB2BGR))
    cv2.imwrite("Dataset/output_colored.jpg", cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR))
#<<<<<<< HEAD

    # Return colored image
#    return cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
#=======
#>>>>>>> 04570266a04f5d1f0f33d914d14411826b15df4e
    #cv2.imwrite("Dataset/target.jpg", cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR))

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

def start_test(opt):

  # load training configuration from yaml file
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
