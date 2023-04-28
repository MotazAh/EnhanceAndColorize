import argparse

from numpy import False_

def refdata_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--op", type=str, required=True, help='Operations: get_features, get_ref')
    parser.add_argument("--img_path", type=str, required=False)
    parser.add_argument("--data_dir", type=str, required=False)
    parser.add_argument("--feature_dir",default='', type=str, required=False)
    parser.add_argument("--ref_dir", type=str, required=False)
    parser.add_argument("--ref_feature_dir", type=str, required=False)
    parser.add_argument("--write_to", type=str, required=False)

    opt = parser.parse_args()
    return opt

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True, help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='', help='Continued training path')
    parser.add_argument("--use_gpu", type=bool, default=False)
    
    opt = parser.parse_args()
    return opt

def test_parser():
  parser = argparse.ArgumentParser(description="synthetic data generation")
  parser.add_argument("--hypes_yaml", type=str, required=True, help='data generation yaml file needed ')
  parser.add_argument('--model_dir', default='', help='Continued training path')
  parser.add_argument("--img_path", type=str, required=True)
  parser.add_argument("--ref_path", type=str, required=True)

  opt = parser.parse_args()
  return opt

def test_blank_parser():
  parser = argparse.ArgumentParser(description="synthetic data generation")
  parser.add_argument("--hypes_yaml", type=str, default='', help='data generation yaml file needed ')
  parser.add_argument('--model_dir', default='', help='Continued training path')
  parser.add_argument("--img_path", type=str, default='')
  parser.add_argument("--ref_path", type=str, default='')

  opt = parser.parse_args()
  return opt