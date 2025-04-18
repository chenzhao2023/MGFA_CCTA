import nibabel as nib
import numpy as np
import cv2
import os
from tqdm import tqdm
from process import get_dilated_contour
def process_slice(slice_2d, kernel_size=25, iterations=1):

    slice_2d[np.where(slice_2d > 0)] = 255

    contour = get_dilated_contour(slice_2d.astype(np.uint8))

    contour[np.where(contour > 0)] = 1

    return contour


def dice_coefficient(pred, target, smooth=1e-6):

    pred_flat = pred.flatten()
    target_flat = target.flatten()

    intersection = np.sum(pred_flat * target_flat)
    dice = (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)

    return dice




def process_3d_image(input_path_vess, input_path, output_path, kernel_size=5, iterations=1):

    img = nib.load(input_path)
    img_data = img.get_fdata()

    img1 = nib.load(input_path_vess)
    img_data1 = img1.get_fdata()

    num_slices = img_data.shape[2]

    processed_data = np.zeros_like(img_data)

    for i in range(num_slices):
        slice_2d = img_data[:, :, i]
        if np.sum(slice_2d) == 0:
            processed_data[:, :, i] = slice_2d
        else:
            slice_2d = cv2.flip(slice_2d, 0)
            processed_slice = process_slice(slice_2d, kernel_size, iterations)
            processed_data[:, :, i] = cv2.flip(processed_slice, 0)

    img_data11 = img_data1 * processed_data
    # print(dice_coefficient(img_data1,img_data11))

    processed_img = nib.Nifti1Image(processed_data, img.affine, img.header)

    nib.save(processed_img, output_path)
    print(f"saved to : {output_path}")


def batch_process(input_path_vess, input_dir, output_dir, kernel_size=5, iterations=1):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith('.nii.gz'):

            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            input_path_v = os.path.join(input_path_vess, file_name.replace("_",''))

            process_3d_image(input_path_v, input_path, output_path, kernel_size=kernel_size, iterations=iterations)


input_dir = r"..\mask"
output_dir = r"..\contour"
input_path_vess = r'..\trainmask'

batch_process(input_path_vess, input_dir, output_dir, kernel_size=25, iterations=1)
