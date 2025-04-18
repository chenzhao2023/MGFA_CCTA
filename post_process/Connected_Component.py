from skimage import measure
import numpy as np




def backpreprcess(ret_box):
    box = []
    [heart_res, num] = measure.label(ret_box, return_num = True)
    region = measure.regionprops(heart_res)
    for i in range(num):
        box.append(region[i].area)
    label_num = box.index(max(box)) + 1

    heart_res[heart_res != label_num] = 0
    heart_res[heart_res == label_num] = 1
    heart_res = np.array(heart_res, dtype = 'uint8')
    return heart_res


def backpreprcess_min(pred,thrd=1000):
    labeled_volume = measure.label(pred, connectivity=3)
    unique_labels, label_counts = np.unique(labeled_volume, return_counts=True)
    min_volume_threshold = thrd

    for labelr, count in zip(unique_labels, label_counts):
        if count < min_volume_threshold:
            pred[labeled_volume == labelr] = 0
    return pred