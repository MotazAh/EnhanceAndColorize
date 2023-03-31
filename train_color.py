import os

from utils.parser import train_parser

import torch
import torch.optim.lr_scheduler as lr_scheduler
import cv2
import numpy as np

from tensorboardX import SummaryWriter
from torchvision.models import resnet34, resnet101
from torchvision.models.resnet import BasicBlock, Bottleneck

from utils import helper, loss
from utils.color_space_convert import lab_to_rgb
from models.networks import AttentionExtractModule
from hypes_yaml import yaml_utils

def train(opt, hypes):
  
  print('loading dataset')
  # Real test is for testing real grayscale old images (Not available currently)
  loader_train, loader_val = helper.create_dataset(hypes,
                                                   train=True,
                                                   real=False)

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
  
  print("Setting criterion")
  criterion = helper.setup_loss(hypes)
  print("Setting optimizer")
  optimizer = helper.setup_optimizer(hypes['train_params']['solver'], model)
  print("Setting scheduler")
  scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

   # load saved model for continue training or train from scratch
  print("Save path: " + opt.model_dir)
  if opt.model_dir:
    print("Loading previous model")
    saved_path = opt.model_dir
    init_epoch, model = helper.load_saved_model(saved_path, model, use_gpu)
  else:
    print("Initializing save model folder")
    # setup saved model folder
    init_epoch = 0
    saved_path = helper.setup_train(hypes)

  print("Setting Summary Writer")
  writer = SummaryWriter(saved_path)

  print('Training Start')
  epoches = hypes['train_params']['epoches']
  step = 0
  for epoch in range(init_epoch, max(epoches, init_epoch)):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        for i, batch_data in enumerate(loader_train):
            # clean up grad first
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_batch, input_l, gt_ab, gt_l, ref_gray, ref_ab, ref_l, input_gray = batch_data['input_image'], \
                                                                   batch_data['input_L'], \
                                                                   batch_data['gt_ab'], batch_data['gt_L'], \
                                                                   batch_data['ref_gray'], batch_data['ref_ab'], \
                                                                   batch_data['ref_l'], batch_data['input_gray']
            #print("ref_gray")
            #print(ref_gray)
            #print("input_batch")
            #print(input_batch)

            if use_gpu:
                input_batch = input_batch.cuda()
                input_l = input_l.cuda()
                gt_ab = gt_ab.cuda()
                gt_l = gt_l.cuda()
                ref_gray = ref_gray.cuda()
                ref_ab = ref_ab.cuda()
                ref_l = ref_l.cuda()
            
            if torch.isnan(input_batch).any():
              print("Found nan in input_batch")
            if torch.isnan(input_l).any():
              print("Found nan in input_l")
            if torch.isnan(gt_ab).any():
              print("Found nan in gt_ab")
            if torch.isnan(gt_l).any():
              print("Found nan in gt_l")
            if torch.isnan(ref_gray).any():
              print("Found nan in ref_gray")
            if torch.isnan(ref_ab).any():
              print("Found nan in ref_ab")
            
            # model inference and loss cal
            out_dict = model(input_l, input_batch, ref_ab, ref_gray, att_model)
            final_loss = loss.loss_sum(hypes, criterion, out_dict, gt_ab)
            print(final_loss)

            

            if (not torch.isnan(final_loss).any()):
              # back-propagation
              final_loss.backward()
              optimizer.step()

            # plot and print training info
            if step % hypes['train_params']['display_freq'] == 0:
                model.eval()

                out_dict = model(input_l, input_batch, ref_ab, ref_gray, att_model)
                out_train = torch.clamp(out_dict['output'], -1., 1.)

                if use_gpu:
                  out_train = lab_to_rgb(input_l, out_train).cuda()
                  target_train = lab_to_rgb(gt_l, gt_ab).cuda()
                  ref_train = lab_to_rgb(ref_l, ref_ab).cuda()

                  output_np = out_train[0].cpu().numpy()
                  target_np = target_train[0].cpu().numpy()
                  ref_np = ref_train[0].cpu().numpy()


                else:
                  out_train = lab_to_rgb(input_l, out_train)
                  target_train = lab_to_rgb(gt_l, gt_ab)
                  ref_train = lab_to_rgb(ref_l, ref_ab)

                  output_np = out_train[0].numpy()
                  target_np = target_train[0].numpy()
                  ref_np = ref_train[0].numpy()
                
                print(out_train.size())

                print(target_np.shape)

                output_np = output_np * 255
                output_np = output_np.transpose(1, 2, 0)

                target_np = target_np * 255
                target_np = target_np.transpose(1, 2, 0)

                ref_np = ref_np * 255
                ref_np = ref_np.transpose(1, 2, 0)

                print("Writing output, target and ref images")

                output_np = np.concatenate((input_gray[0], ref_np, output_np, target_np), 1)

                cv2.imwrite("Dataset/output.jpg", cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR))
                cv2.imwrite("Dataset/target.jpg", cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR))

                psnr_train = loss.batch_psnr(out_train, target_train, 1.)
                print("[epoch %d][%d/%d], total loss: %.4f, PSNR: %.4f" % (epoch + 1, i + 1, len(loader_train),
                                                                           final_loss.item(), psnr_train))
                writer.add_scalar('generator pretrain loss', final_loss.item(), step)
                writer.add_scalar('PSNR during pretrain', psnr_train, step)
            step += 1

        # log images

        writer = helper.log_images(input_l, input_batch, ref_ab, ref_gray, writer, model, epoch, att_model, use_gpu)
        
        # evaluate model on validation dataset
        if epoch % hypes['train_params']['eval_freq'] == 0:
            print("Evaluating model on validation set")
            writer = helper.val_eval(model, att_model, loader_val, writer, opt, epoch, use_gpu)
        
        #Save model
        if epoch % hypes['train_params']['writer_freq'] == 0:
            torch.save(model.state_dict(), os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

  return model, att_model, writer


  pass

if __name__ == '__main__':
  # load training configuration from yaml file
  opt = train_parser()
  print(opt.hypes_yaml)
  hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
  opt.use_gpu = hypes["train_params"]["use_gpu"]
  # Check gpu
  #opt.use_gpu
  if opt.use_gpu:
    print("Using gpu")
  else:
    print("Not using gpu")
  
  print("Starting")
  train(opt, hypes)