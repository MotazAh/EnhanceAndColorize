from get_ref import *
from test_color import start_test
from utils.parser import test_parser, test_blank_parser

class Opt:
  hypes_yaml = ""
  model_dir = ""
  img_path = ""
  ref_path = ""

def colorize_img(c_input_image_path="input_image.jpg", c_ref_path="best_ref.jpg", c_model_dir="/content/drive/MyDrive/saved_model", c_hypes="hypes_yaml/config.yaml"):
  opt = Opt()
  opt.use_gpu = True
  opt.hypes_yaml = c_hypes
  opt.model_dir = c_model_dir
  opt.img_path = c_input_image_path
  opt.ref_path = c_ref_path
  get_top()
  colored_img = start_test(opt)
  print("DONE")
  return colored_img

def get_top(ref_feature_dir="Dataset/feature_vectors", ref_dir="Dataset/test2017", img_path="input_image.jpg"):
  feature_reader(ref_feature_dir, ref_dir, False, False)
  print("Finding top 2 images")
  top = find_top_image(img_path, verbose=True)[0]