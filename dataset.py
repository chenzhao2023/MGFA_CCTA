from torch.utils.data import Dataset
import os
import torch
import numpy as np
import nibabel as nib
import re


def extract_number_mask(filename):
    match = re.search(r'(\d+)_(\d+)_label', filename)
    if match:
        group = match.group(1)
        number = int(match.group(2))
        return group, number
    return None, -1


def extract_number_img(filename):
    match = re.search(r'(\d+)_(\d+)_img', filename)
    if match:
        group = match.group(1)
        number = int(match.group(2))
        return group, number
    return None, -1


class VesselsSegmentionDataSetsave_t(Dataset):
    def __init__(self, rootdir, rate):
        self.root_dir = rootdir
        self.imgdir = os.path.join(rootdir, r'data')
        self.maskdir = os.path.join(rootdir, r'mask')
        self.contoursdir = os.path.join(rootdir, r'contours')

        self.patients = [i for i in os.listdir(self.imgdir) if i.endswith(r'.nii.gz')]
        self.patientsmask = [i for i in os.listdir(self.maskdir) if i.endswith(r'.nii.gz')]
        self.patientsmaskcontours = [i for i in os.listdir(self.contoursdir) if i.endswith(r'.nii.gz')]

        self.patients = sorted(self.patients, key=extract_number_img)
        self.patientsmask = sorted(self.patientsmask, key=extract_number_mask)
        self.patientsmaskcontours = sorted(self.patientsmaskcontours, key=extract_number_mask)

        length = int(len(self.patients))
        self.patients = self.patients[:int(rate * length)]
        self.patientsmask = self.patientsmask[:int(rate * length)]
        self.patientsmaskcontours = self.patientsmaskcontours[:int(rate * length)]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        name = self.patients[item]
        image = nib.load(os.path.join(self.imgdir, self.patients[item])).get_fdata()
        label = nib.load(os.path.join(self.maskdir, self.patientsmask[item])).get_fdata()
        cou = nib.load(os.path.join(self.contoursdir, self.patientsmaskcontours[item])).get_fdata()

        if np.max(image) > 1:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))


        image = np.array(image)
        label = np.array(label)
        cou = np.array(cou)

        label = torch.from_numpy(label)
        data = torch.from_numpy(image)
        cou = torch.from_numpy(cou)

        if len(image.shape) == 3:
            data = data.unsqueeze(0).to(torch.float32)
            label = label.unsqueeze(0).to(torch.float32)
            cou = cou.unsqueeze(0).to(torch.float32)
        else:
            data = data.to(torch.float32)

        data = torch.cat([data, cou], dim=0)

        return (data, label, name)


class VesselsSegmentionDataSetsave_te(Dataset):
    def __init__(self, rootdir, rate):
        self.root_dir = rootdir
        self.imgdir = os.path.join(rootdir, r'data')
        self.maskdir = os.path.join(rootdir, r'mask')

        self.patients = [i for i in os.listdir(self.imgdir) if i.endswith(r'.nii.gz')]
        self.patientsmask = [i for i in os.listdir(self.maskdir) if i.endswith(r'.nii.gz')]

        self.patients = sorted(self.patients, key=extract_number_img)
        self.patientsmask = sorted(self.patientsmask, key=extract_number_mask)

        length = int(len(self.patients))
        self.patients = self.patients[:int(rate * length)]
        self.patientsmask = self.patientsmask[:int(rate * length)]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = nib.load(os.path.join(self.imgdir, self.patients[item])).get_fdata()
        label = nib.load(os.path.join(self.maskdir, self.patientsmask[item])).get_fdata()

        if np.max(image) > 1:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))


        name = self.patients[item].split('.')[0]

        image = np.array(image)
        label = np.array(label)

        label = torch.from_numpy(label)
        data = torch.from_numpy(image)

        if len(image.shape) == 3:
            data = data.unsqueeze(0).to(torch.float32)
            label = label.unsqueeze(0).to(torch.float32)
        else:
            data = data.to(torch.float32)

        return (data, label, name)



class VesselsSegmentionDataSet512(Dataset):
    def __init__(self,test_list):

        data_dir = r'..\train'
        mask_dir = r'..\trainmask'

        cou_dir = r'..\contours'

        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.patients = test_list
        self.cou_dir = cou_dir

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        image = nib.load(os.path.join(self.data_dir, str(self.patients[item]) + 'img.nii.gz')).get_fdata()
        if np.max(image)>1:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        cou = nib.load(os.path.join(self.cou_dir, str(self.patients[item]) + '_label.nii.gz')).get_fdata()
        label = nib.load(os.path.join(self.mask_dir, str(self.patients[item]) + 'label.nii.gz'))
        labels_header = label.header
        label = label.get_fdata()

        label[np.where(label > 0)] = 1
        cou[np.where(cou > 0)] = 1


        # image, label, cou = lrv(image, label,cou, heart_data)
        # image, label, cou = data3(image, label, cou)
        shape = image.shape
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        cou = torch.from_numpy(cou)
        image = np.transpose(image, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))
        cou = np.transpose(cou, (2, 0, 1))
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)
        cou = cou.unsqueeze(0)
        name = self.patients[item]
        pixdim = labels_header.get("pixdim")
        spacing = (pixdim[3], pixdim[1], pixdim[2])


        return (image, label,cou, name, shape, spacing)
