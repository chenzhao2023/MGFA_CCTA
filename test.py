import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import warnings
from skimage import measure
from scipy.ndimage import binary_erosion
from metrics import calculate_precision_recall
import pandas as pd
warnings.filterwarnings("ignore")
from dataset import *
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from models import unet3d, MGFA
from tqdm import tqdm
from utils.crop_utils import *
import nibabel as nib
import random
from metrics import compute_dice_coefficientnp
from metrics import calculate_hd95
# from post_process.Connected_Component import backpreprcess_min
def backpreprcess_min(pred,thrd=1000):
    labeled_volume = measure.label(pred, connectivity=3)
    unique_labels, label_counts = np.unique(labeled_volume, return_counts=True)
    min_volume_threshold = thrd

    for labelr, count in zip(unique_labels, label_counts):
        if count < min_volume_threshold:
            pred[labeled_volume == labelr] = 0
    return pred

def compute_dice_coefficient(seg_true, seg_pred, smooth=1e-5):
    intersection = torch.sum(seg_true * seg_pred)
    union = torch.sum(seg_true) + torch.sum(seg_pred)

    # 计算 Dice 系数
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
    return dice_coeff


def compute_sensitivity(seg_true, seg_pred):
    true_positive = np.sum(seg_true * seg_pred)
    false_negative = np.sum(seg_true * (1 - seg_pred))
    sensitivity = true_positive / (true_positive + false_negative)
    return sensitivity


def compute_specificity(seg_true, seg_pred):
    true_negative = np.sum((1 - seg_true) * (1 - seg_pred))
    false_positive = np.sum((1 - seg_true) * seg_pred)
    specificity = true_negative / (true_negative + false_positive)
    return specificity



def custom_round_array(array):
    result = torch.empty_like(array, device='cuda')
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                value = array[i, j, k]
                integer_part = int(value)
                decimal_part = value - integer_part
                if decimal_part < 0.4:
                    result[i, j, k] = integer_part
                else:
                    result[i, j, k] = integer_part + 1
    return result




def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file.readlines()]
    return data

def custom_round_array(array):
    result = np.empty_like(array)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                value = array[i, j, k]
                integer_part = int(value)
                decimal_part = value - integer_part
                if decimal_part < 0.4:
                    result[i, j, k] = integer_part
                else:
                    result[i, j, k] = integer_part + 1
    return result





if __name__ == '__main__':

    overlap = [16, 20, 20] #[32 40 40], [64 80 80]
    patch_size = [128, 160, 160]
    batch_size = 1
    seed = 58790
    num_workers = 1
    save_dir = r''
    class_thrd = 0.5
    dice_coefficient_value_avg_te = 0
    sensitivity_value_avg_te = 0
    specificity_value_avg_te = 0
    dice_org = 0
    dice_back = 0
    all_precision = 0
    all_recall = 0
    all_hsd = 0
    hs_value_avg_te = 0
    num = 0
    device = torch.device("cuda")

    net = MGFA.MGFANet().to(device)
    checkpoint = torch.load(r'', map_location=device)
    net = nn.DataParallel(net)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)


    #0.027 0.1054 0.2930 0.1257


    torch.manual_seed(seed)
    random.seed(seed)
    # data_all = os.listdir(r'')
    test_list = read_data_from_file('data/test.txt')
    test_set = VesselsSegmentionDataSet512(test_list)

    print('READY')


    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)  # 获取测试集数据加载器

    for m in net.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm3d:
                child.track_running_stats = False
                child.running_mean = None
                child.running_var = None

    net.eval()
    for m in net.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.track_running_stats = False

    count = 0
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm3d):
            count += 1
            if count >= 2:
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    for i, (inpute, labele, cous, name, shapes,spacing) in tqdm(enumerate(test_loader)):


        patchs_input = crop_volume_reverse_uniform(inpute, patch_size, overlap)
        mask_merged_volume = labele.cpu().detach().numpy()
        patchs_cous = crop_volume_reverse_uniform(cous, patch_size, overlap)
        out = np.zeros((len(patchs_input), 128, 160, 160))
        for p in range(len(patchs_input)):

            inpute = patchs_input[p].float().to(device)
            cou = patchs_cous[p].float().to(device)
            outputs_te = net(inpute, cou)
            outputs_te = torch.sigmoid(outputs_te)
            out[p] = outputs_te.cpu().detach().numpy()


        pred_merged_volume = assemble_patchesnp(out, shapes, patch_size, overlap)


        pred_merged_volume[pred_merged_volume > class_thrd] = 1
        pred_merged_volume[pred_merged_volume < class_thrd] = 0
        # pred_merged_volume = np.argmax(pred_merged_volume, axis=0)

        diceo = compute_dice_coefficientnp(mask_merged_volume, pred_merged_volume)
        dice_org += diceo

        # mask_merged_volume = mask_merged_volume.cpu().detach().numpy()
        pred_back = backpreprcess_min(pred_merged_volume,thrd=3000)
        # pred_back = pred_merged_volume
        diceback = compute_dice_coefficientnp(mask_merged_volume, pred_back)
        dice_back += diceback
        num = num + 1

        precision, recall = calculate_precision_recall(pred_back, mask_merged_volume)
        hsd = calculate_hd95(pred_back,mask_merged_volume[0,0],spacing)

        all_precision += precision
        all_recall += recall
        all_hsd += hsd

        print(" ")
        print(f"Dice_org: {diceo:.5f}, ALL Dice_org:{dice_org / num:.5f}")
        print(f"Dice_Back: {diceback:.5f}, ALL Dice_Back:{dice_back / num:.5f}")

        print('-------------------------------------------------------------------------------')

