import glob
import os
import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
from tqdm import tqdm
import torch
from utils.crop_utils import crop_volume_reverse_uniform
import nibabel as nib
import shutil


def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file.readlines()]
    return data

def crop_data(datalist,data_dir , savedir, patch_size, overlap, thrd, if_sele=False):


    train_mask_path = os.path.join(savedir, 'mask')
    train_data_path = os.path.join(savedir, 'data')

    if not os.path.exists(train_mask_path):
        os.makedirs(train_mask_path)
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    for i in tqdm(datalist):
        num = 0
        i = str(i)
        i = i[:8]
        image = nib.load(os.path.join(data_dir, str(i), 'img.nii.gz')).get_fdata()
        label = nib.load(os.path.join(data_dir, str(i), 'label.nii.gz'))
        print(f'shape: {image.shape}')
        labels_header = label.header
        label = label.get_fdata()

        label[np.where(label > 0)] = 1

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        image = np.transpose(image, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)
        patchs_input = crop_volume_reverse_uniform(image, patch_size, overlap)
        patchs_label = crop_volume_reverse_uniform(label, patch_size, overlap)
        print(f'Number of patches: {len(patchs_input)}')
        print(f'Patch shape: {patchs_input[0].shape}')
        if not patchs_input[0].shape[2:] == tuple(patch_size):
            print("___error___")
            continue

        for j in range(len(patchs_input)):
            input_tr = patchs_input[j].cpu()
            label_tr = patchs_label[j].cpu()
            input_tr = input_tr.squeeze(0).squeeze(0)
            label_tr = label_tr.squeeze(0).squeeze(0)

            if not if_sele:
                a = np.prod(label_tr.shape) * thrd
                if label_tr.sum() < a:
                    # print(label_tr.sum(), a)
                    num += 1
                    continue
            modified_nifti_image = nib.Nifti1Image(input_tr, None, header=labels_header)
            nib.save(modified_nifti_image, savedir + '/' + 'data/' + i + f'_{j}_image.nii.gz')
            modified_nifti_image = nib.Nifti1Image(label_tr, None, header=labels_header)
            nib.save(modified_nifti_image, savedir + '/' + 'mask/' + i + f'_{j}_label.nii.gz')
        print(num)

def copy_patch(lists, redirs, dedirs):
    all_patch = glob.glob(os.path.join(redirs, '**', '*.nii.gz'), recursive=True)

    train_mask_path = os.path.join(dedirs, 'mask')
    train_data_path = os.path.join(dedirs, 'data')
    train_cou_path = os.path.join(dedirs, 'contours')
    if not os.path.exists(train_mask_path):
        os.makedirs(train_mask_path)
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    if not os.path.exists(train_cou_path):
        os.makedirs(train_cou_path)

    for i in tqdm(lists):

        patch_list = [s for s in all_patch if str(i) in s]

        for j in patch_list:

            if 'mask' in j:
                shutil.copy(j, os.path.join(train_mask_path, os.path.basename(j)))
            elif 'contours' in j:
                shutil.copy(j, os.path.join(train_cou_path, os.path.basename(j)))
            elif 'img' in j:
                shutil.copy(j, os.path.join(train_data_path, os.path.basename(j)))





if __name__ == '__main__':

    train_list = read_data_from_file('train_data.txt')
    validation_list = read_data_from_file('validation_data.txt')
    test_list = read_data_from_file('test_data.txt')
    overlap = [32, 40, 40]
    patch_size = [128, 160, 160]
    thrd = 0.002 # 0.012 #0.004

    source_datapath = r'' # data dir
    den_traindir = r'data/train'
    den_valdir = r'data/val'



    crop_data(train_list, source_datapath,den_traindir,patch_size,overlap,thrd,False)
    crop_data(validation_list, source_datapath, den_valdir, patch_size, overlap, thrd, False)









