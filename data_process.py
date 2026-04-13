import glob
import os
import warnings

import train_cfg

warnings.filterwarnings("ignore")
import random
import numpy as np
from tqdm import tqdm
import torch
from utils.crop_utils import crop_volume_reverse_uniform
import nibabel as nib
import shutil
from train_cfg import datapro as dp



def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file.readlines()]
    return data

def crop_data(datalist,data_dir , savedir, patch_size, overlap, thrd, if_sele=False, cou_dir=None):


    train_mask_path = os.path.join(savedir, 'mask')
    train_data_path = os.path.join(savedir, 'data')
    contour_data_path = os.path.join(savedir, 'contours_noerode')

    if not os.path.exists(train_mask_path):
        os.makedirs(train_mask_path)
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    if not os.path.exists(contour_data_path):
        os.makedirs(contour_data_path)

    for i in tqdm(datalist):
        num = 0
        i = str(i)
        i = i[:8]
        image = nib.load(os.path.join(data_dir, str(i), 'img.nii.gz')).get_fdata()
        label = nib.load(os.path.join(data_dir, str(i), 'label.nii.gz'))
        contour = nib.load(os.path.join(cou_dir, str(i)+'_label.nii.gz')).get_fdata()
        print(f'shape: {image.shape}')
        labels_header = label.header
        label = label.get_fdata()

        label[np.where(label > 0)] = 1
        contour[np.where(contour > 0)] = 1

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        contour = torch.from_numpy(contour)
        image = np.transpose(image, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))
        contour = np.transpose(contour, (2, 0, 1))
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)
        contour = contour.unsqueeze(0)
        contour = contour.unsqueeze(0)
        patchs_input = crop_volume_reverse_uniform(image, patch_size, overlap)
        patchs_label = crop_volume_reverse_uniform(label, patch_size, overlap)
        patchs_contour = crop_volume_reverse_uniform(contour, patch_size, overlap)
        print(f'Number of patches: {len(patchs_input)}')
        print(f'Number of patches: {len(patchs_label)}')
        print(f'Patch shape: {patchs_input[0].shape}')
        if not patchs_input[0].shape[2:] == tuple(patch_size):
            print("___error___")
            continue
        # asm = assemble_patches(patchs_input,[image.shape[3],image.shape[4],image.shape[2]], patch_size, overlap)
        # print(asm.shape)
        # modified_nifti_image = nib.Nifti1Image(asm.cpu(), None, header=labels_header)
        # nib.save(modified_nifti_image, savedir + '/' + 'data/' + i + f'asm.nii.gz')
        for j in range(len(patchs_input)):
            input_tr = patchs_input[j].cpu()
            label_tr = patchs_label[j].cpu()
            contour_tr = patchs_contour[j].cpu()
            input_tr = input_tr.squeeze(0).squeeze(0)
            label_tr = label_tr.squeeze(0).squeeze(0)
            contour_tr = contour_tr.squeeze(0).squeeze(0)

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
            modified_nifti_image = nib.Nifti1Image(contour_tr , None, header=labels_header)
            nib.save(modified_nifti_image, savedir + '/' + 'contours_noerode/' + i + f'_{j}_label.nii.gz')

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
            elif 'image' in j:
                shutil.copy(j, os.path.join(train_data_path, os.path.basename(j)))





def dataprocess():
    # train_list = read_data_from_file('../data/train.txt')
    # validation_list = read_data_from_file('../data/val.txt')
    # test_list = read_data_from_file('../data/test.txt')
    overlap = [32, 40, 40]
    patch_size = [128, 160, 160]
    thrd = 0.002  # 0.012 #0.004
    seed = 58790

    source_datapath = dp.source_datapath  # data dir
    den_traindir = dp.den_traindir
    den_valdir = dp.den_valdir
    cou_dir = dp.cou_dir

    data_all = os.listdir(source_datapath)

    random.seed(seed)
    test_list = random.sample(data_all, 200)
    train_set = set(data_all) - set(test_list)
    train_list = np.sort(list(train_set)[:700])
    # print(train_list)
    validation_list = list(train_set)[700:]
    # print(test_list)

    crop_data(train_list, source_datapath, den_traindir, patch_size, overlap, thrd, False, cou_dir)
    crop_data(validation_list, source_datapath, den_valdir, patch_size, overlap, thrd, False, cou_dir)


if __name__ == '__main__':

    # train_list = read_data_from_file('../data/train.txt')
    # validation_list = read_data_from_file('../data/val.txt')
    # test_list = read_data_from_file('../data/test.txt')
    overlap = [32, 40, 40]
    patch_size = [128, 160, 160]
    thrd = 0.002 # 0.012 #0.004
    seed = dp.seed

    source_datapath = dp.source_datapath # data dir
    den_traindir = dp.den_traindir
    den_valdir = dp.den_valdir
    cou_dir = dp.cou_dir

    data_all = os.listdir(source_datapath)

    random.seed(seed)
    test_list = random.sample(data_all, 200)
    train_set = set(data_all) - set(test_list)
    train_list = np.sort(list(train_set)[:700])
    # print(train_list)
    validation_list = list(train_set)[700:]
    # print(test_list)




    crop_data(train_list, source_datapath,den_traindir,patch_size,overlap,thrd,False, cou_dir)
    crop_data(validation_list, source_datapath, den_valdir, patch_size, overlap, thrd, False, cou_dir)









