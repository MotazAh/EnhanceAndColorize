import argparse

def refdata_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--op", type=str, required=True, help='Operations: get_features, get_ref')
    parser.add_argument("--img_path", type=str, required=False)
    parser.add_argument("--data_dir", type=str, required=False)
    parser.add_argument("--feature_dir", type=str, required=False)

    opt = parser.parse_args()
    return opt