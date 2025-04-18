import pandas as pd
# import numpy as np
from scipy.spatial import distance
import ast
import os
import nibabel as nib
import re
import cupy as np
import cupy
from tqdm import tqdm

def get_mse(bg, be, st):
    bg /= 100
    be /= 100
    # be += 1
    # bg += 1
    st += 1
    st *= 0.25

    armse = (be - bg) ** 2
    rrmse = ((be - bg) / st) ** 2

    return armse, rrmse

def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file.readlines()]
    return data
def parse_position(pos_str):

    formatted_str = re.sub(r'\s+', ',', pos_str.strip())
    formatted_str = formatted_str.strip(',')
    if formatted_str[1:].startswith(','):
        formatted_str = formatted_str.replace('[,', '[')

    formatted_str = re.sub(r',+', ',', formatted_str)

    return np.array(ast.literal_eval(formatted_str))
def parse_array_string(array_str):
    numbers = re.findall(r"array\((\d+),", array_str)

    return np.array([int(num) for num in numbers])

def calculate_min_distance_and_position_np(position, vessel_points):
    position = position.get()
    vessel_points = vessel_points.get()
    distances = distance.cdist(position, [vessel_points])
    min_index = distances.argmin()
    min_distance = distances.min()
    closest_point = position[min_index]
    return min_distance, closest_point, min_index


def calculate_min_distance_and_position(position, vessel_points):

    distances = cupy.linalg.norm(position - vessel_points, axis=1)


    min_index = distances.argmin()
    min_distance = distances[min_index]
    closest_point = position[min_index]

    return min_distance.item(), closest_point, min_index.item()

def find_points(seg_dt_pos, seg_dt_st, seg_dt_per, label_pt, vessel_dt, dis=20):
    pre_pt = []
    pre_st = []
    pre_percent = []

    for index, i in enumerate(seg_dt_pos):

        if vessel_dt[i[0], i[1], i[2]] == 1:
            pre_pt.append(i)
            pre_st.append(int(seg_dt_st[index]))
            pre_percent.append(int(seg_dt_per[index]))
    if len(pre_pt) == 0:
        seg_dt_pos = np.vstack(seg_dt_pos.to_numpy())
        other_mds, other_pt, min_index = calculate_min_distance_and_position(seg_dt_pos, label_pt)
        print(other_mds)
        if other_mds <= dis:
            return other_pt, seg_dt_st[min_index], seg_dt_per[min_index]
        else:
            return None, None, None
    else:
        pre_pt = np.array(pre_pt)
        mindis, pre_pt, min_index = calculate_min_distance_and_position(pre_pt, label_pt)
        print(mindis)
        if mindis <= dis:
            return pre_pt, pre_st[min_index], pre_percent[min_index]
        else:
            # return None, None
            seg_dt_pos = np.vstack(seg_dt_pos.to_numpy())
            other_mds, other_pt, min_index = calculate_min_distance_and_position(seg_dt_pos, label_pt)
            print(other_mds)
            if other_mds <= dis:
                return other_pt, seg_dt_st[min_index], seg_dt_per[min_index]
            else:
                return None, None, None

def evaluation(label_dir, seg_dir):


    all_acc = 0
    all_acc2 = 0
    num = 0
    st_name = ['minimal', 'mild', 'moderate', 'severe']
    stenosis_all_label = np.zeros(4)
    stenosis_pred_label = np.zeros(4)
    stenosis_all_seg = np.zeros(4)
    stenosis_pred_seg = np.zeros(4)

    armse_all = np.zeros(5)
    rrmse_all = np.zeros(5)

    TP = 0
    FN = 0
    FP = 0

    data_list = os.listdir(seg_dir)


    for i in tqdm(data_list):
        print(i)
        succ_num = 0



        seg_data = pd.read_excel(os.path.join(seg_dir, i, 'st_area_data.xlsx'), sheet_name='Sheet1')
        label_data = pd.read_excel(os.path.join(label_dir, i, 'st_area_data.xlsx'), sheet_name='Sheet1')
        if 'array' in seg_data['position'][0]:

            seg_data['position'] = seg_data['position'].apply(parse_array_string)
        else:

            seg_data['position'] = seg_data['position'].apply(parse_position)
        label_data['position'] = label_data['position'].apply(parse_position)
        if label_data.shape[0]==0:
            continue
        for s in range(label_data.shape[0]):
            stenosis_all_label[int(label_data['stenosis'][s])] += 1
        for s in range(seg_data.shape[0]):
            stenosis_all_seg[int(seg_data['stenosis'][s])] += 1

        for j in range(label_data.shape[0]):
            vessel_num = label_data.iloc[j]['vessel_num']
            vessel_data = nib.load(os.path.join(label_dir, i, rf'vessel_patch{vessel_num}.nii.gz')).get_fdata()
            vessel_data = cupy.asarray(vessel_data)
            vessel_data[np.where(vessel_data > 0)] = 1
            match_pt, sten, percent = find_points(seg_data['position'], seg_data['stenosis'], seg_data['percent'], label_data['position'][j], vessel_data)
            if match_pt is not None:
                stenosis_pred_label[int(label_data['stenosis'][j])] += 1
                stenosis_pred_seg[sten] += 1
                armse, rrmse = get_mse(int(label_data['percent'][j]), percent, int(label_data['stenosis'][j]))
                armse_all[sten] += armse
                rrmse_all[sten] += rrmse
                armse_all[4] += armse
                rrmse_all[4] += rrmse

                succ_num += 1

        TP+=succ_num
        FP+= label_data.shape[0] - succ_num
        FN+= seg_data.shape[0] - succ_num
        num += 1
        acc = succ_num / label_data.shape[0]
        acc2 = succ_num / seg_data.shape[0]
        all_acc += acc
        all_acc2 += acc2

        print(succ_num/label_data.shape[0],succ_num/seg_data.shape[0])



        for s in range(4):
            if stenosis_all_label[s] == 0:
                print('TPR:', st_name[s], 'None')
            else:
                print('TPR:', st_name[s], stenosis_pred_label[s] / stenosis_all_label[s])
                # print(stenosis_pred_label[s],'/',stenosis_all_label[s])
        for s in range(4):
            if stenosis_all_label[s] == 0:
                print('PPV:', st_name[s], 'None')
            else:
                print('PPV:', st_name[s], stenosis_pred_seg[s] / stenosis_all_label[s])
                # print(stenosis_pred_seg[s], '/', stenosis_all_label[s])
        for s in range(4):
            if stenosis_pred_label[s] == 0:
                print('ARMSE:', st_name[s], 'None')
            else:
                print('ARMSE:', st_name[s], np.sqrt(armse_all[s] / stenosis_pred_label[s]))
                # print(armse_all[s], '/', stenosis_pred_label[s])
        for s in range(4):
            if stenosis_pred_label[s] == 0:
                print('RRMSE:', st_name[s], 'None')
            else:
                print('RRMSE:', st_name[s], np.sqrt(rrmse_all[s] / stenosis_pred_label[s]))
                # print(rrmse_all[s], '/', stenosis_pred_label[s])

        print('------------', acc, acc2, all_acc / num, all_acc2 / num)
        print('------------',TP/(TP+FN), TP/(TP+FP))
    print(TP,FP,FN)
    print(np.sqrt(armse_all[4]/TP), np.sqrt(rrmse_all[4]/TP))

if __name__ == '__main__':
    label_dir = r''
    seg_dir = r''

    evaluation(label_dir, seg_dir)