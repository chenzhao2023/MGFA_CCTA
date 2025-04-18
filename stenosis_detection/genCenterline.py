import os

from skimage import measure
from skimage.morphology import skeletonize_3d
import nibabel as nib
import numpy as np
from skimage.measure import label
import networkx as nx

def create_segment_mask(image_shape, segment):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for point in segment:
        x, y, z = map(int, point)
        mask[x, y, z] = 1
    return mask


def crop_centerline(patient,data_dir, predict=False):



    label_path = os.path.join(data_dir, patient,'label.nii.gz')
    if predict:
        label_path = os.path.join(data_dir, patient, 'predaf.nii.gz')
    centerline_path = os.path.join(data_dir, patient,'centerline.nii.gz')
    output_centerline_path = os.path.join(data_dir,patient,'degree2.nii.gz')

    arrays = nib.load(label_path)


    affine = arrays.affine
    arrays = arrays.get_fdata()
    arrays = np.array(arrays)


    # arrays[np.where(arrays > 0.)] = 1


    skeleton = skeletonize_3d(arrays)
    nib.save(nib.Nifti1Image(skeleton, affine), os.path.join(data_dir, patient, 'centerlines.nii.gz'))

    # nib.save(nib.Nifti1Image(skeleton, affine), os.path.join(data_dir,patient,'centerline.nii.gz'))


    # skeleton = skeletonize_3d(skeleton)
    #
    # nib.save(nib.Nifti1Image(skeleton, affine), output_centerline_path)


    graph = nx.Graph()
    for index in np.argwhere(skeleton):
        graph.add_node(tuple(index))
    branch_points = []
    branch_points2 = []
    for index in np.argwhere(skeleton):
        x, y, z = index
        num = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:

                    neighbor = (x + dx, y + dy, z + dz)
                    if graph.has_node(neighbor):
                        num = num + 1
                        graph.add_edge((x, y, z), neighbor)

        if num == 3:
            branch_points.append(index)
        if num == 2:
            branch_points2.append(index)

    # branch_points = [node for node, degree in dict(graph.degree()).items() if degree == 1]
    print(len(branch_points))
    print(len(branch_points2))



    mask = create_segment_mask(arrays.shape, branch_points2)
    nib.save(nib.Nifti1Image(mask, affine), os.path.join(data_dir,patient,'degree2.nii.gz'))
    mask = create_segment_mask(arrays.shape, branch_points)

    box = []
    [heart_res, num] = measure.label(mask, return_num = True)
    region = measure.regionprops(heart_res)
    for i in range(num):
        box.append(region[i].area)
    for i, reg in enumerate(region):
        if box[i] <= 20:
            continue
        mask_i = np.zeros_like(mask)
        mask_i[heart_res == i+1] = True
        nib.save(nib.Nifti1Image(mask_i, affine), os.path.join(data_dir,patient,'ctl_patch{}.nii.gz'.format(i)))

    print(num)

    # nib.save(nib.Nifti1Image(mask, affine), output_centerline_path)







