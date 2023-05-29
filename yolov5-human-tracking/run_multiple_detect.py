import numpy as np
# from sklearn.model_selection import train_test_split
import random
import os
# import cv2
import argparse
# import shutil
# from PIL import Image

random.seed(0)


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--node_id', type=str, default='05',
                        help='node_id')
    parser.add_argument('--data_folder', type=str, default='label_data',
                        help='data_folder')
    parser.add_argument('--start_sample_id', type=int, default=0,
                        help='start_sample_id')
    parser.add_argument('--end_sample_id', type=int, default=10,
                        help='end_sample_id')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device_id')

    opt = parser.parse_args()

    # opt.load_folder = "../../../label_data/V0{}/all_data/".format(opt.node_id)
    # opt.save_folder = "./V0{}_data_selection/depth_images/".format(opt.node_id)
    # if not os.path.isdir(opt.index_path):
    # 	os.makedirs(opt.index_path)


    return opt


def main():

    opt = parse_option()

    # label_1 = np.load(opt.load_folder + "label_1.npy")

    # num_of_sample = label_1.shape[0]

    # print("num_of_sample:", num_of_sample)

    for sample_idx in range(opt.start_sample_id, opt.end_sample_id):
        print("Yolo detect sample:", sample_idx)
        os.system("python3 detect.py --source ../../../{}/V0{}/depth_images/sample_{}/ --name V0{}_{}/exp{} --device {}".format(opt.data_folder, opt.node_id, sample_idx, opt.node_id, opt.data_folder, sample_idx+1, opt.device_id))



if __name__ == '__main__':
    main()
