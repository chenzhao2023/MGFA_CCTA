import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from genCSA import get_stenosis
from CSA import crop_vessel
from genCenterline import crop_centerline
from tqdm import tqdm
import cv2
import nibabel as nib
import numpy as np
from evaluation import evaluation
from visual_compare import visual_main

def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file.readlines()]
    return data
if __name__ == '__main__':

    data_dir = r'samples'
    data_list = os.listdir(data_dir)

    for i in tqdm(data_list):


        print(i)

        crop_centerline(i, data_dir, predict=False)
        crop_vessel(i, data_dir, predict=False)
        get_stenosis(data_dir, i)

    visual_main(data_dir, data_list[2])


