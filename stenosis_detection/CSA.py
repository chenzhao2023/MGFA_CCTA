import numpy
import nibabel as nib
from sort_ctl import find_endpoints, sort_centerline
from scipy.ndimage import label
import os
from tqdm import tqdm
import cupy as np
import cupy

def extract_connected_region(labels, center, crop=False):

    if crop:
        labeled_array, num_features = label(labels.get())
    else:
        labeled_array, num_features = label(labels.get(), structure=numpy.ones((3, 3, 3)))

    x, y, z = center
    labeled_array = np.array(labeled_array)
    center_label = labeled_array[x, y, z]

    connected_region = np.argwhere(labeled_array == center_label)

    return connected_region, num_features


def get_direction_vector(centerline, index, k=3):

    start_index = index
    end_index = min(len(centerline) - 1, index + k)
    start_point = cupy.array(centerline[start_index])
    end_point = cupy.array(centerline[end_index])
    direction_vector = end_point - start_point
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    return direction_vector


def extract_cross_section_new(label, center, direction_vector, max_radius=20, thrd=0.5):

    W, H, D = label.shape
    x_min, x_max = max(0, center[0] - max_radius), min(W, center[0] + max_radius)
    y_min, y_max = max(0, center[1] - max_radius), min(H, center[1] + max_radius)
    z_min, z_max = max(0, center[2] - max_radius), min(D, center[2] + max_radius)
    x_min = int(x_min)
    x_max = int(x_max)
    y_min = int(y_min)
    y_max = int(y_max)
    z_min = int(z_min)
    z_max = int(z_max)
    x, y, z = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), np.arange(z_min, z_max), indexing='ij')
    coords = np.stack([x, y, z], axis=-1)
    coords = np.array(coords)
    center = np.array(center)
    relative_coords = coords - center

    dot_product = np.dot(relative_coords.reshape(-1, 3), direction_vector)
    mask = np.abs(dot_product) < thrd

    mask = mask.reshape(x_max - x_min, y_max - y_min, z_max - z_min)
    label_cropped = label[x_min:x_max, y_min:y_max, z_min:z_max]
    label_cropped = cupy.asarray(label_cropped)
    cross_section_points = np.argwhere(mask & (label_cropped == 1))
    cross_section_points += cupy.array([x_min, y_min, z_min])
    areas = len(cross_section_points)

    return cross_section_points, areas


def get_plane_normal_vector(direction_vector):

    for i in range(len(direction_vector)):
        if direction_vector[i] != 0:
            normal_vector = np.zeros_like(direction_vector)
            normal_vector[i] = -direction_vector[(i + 1) % len(direction_vector)]
            normal_vector[(i + 1) % len(direction_vector)] = direction_vector[i]
            return normal_vector / np.linalg.norm(normal_vector)
    return np.cross(direction_vector, np.array([1, 0, 0]))


def extract_cross_section(label, center, direction_vector, plane_normal, max_radius=30):

    x, y, z = map(int, center)
    radius = max_radius
    points_in_plane = []

    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            for k in range(-radius, radius + 1):
                point = center + i * direction_vector + j * plane_normal
                xi, yi, zi = map(int, point)
                if 0 <= xi < label.shape[0] and 0 <= yi < label.shape[1] and 0 <= zi < label.shape[2]:
                    if label[xi, yi, zi] == 1:
                        points_in_plane.append([xi, yi, zi])

    return points_in_plane


def stable_areas(binary_label, centerline, ini_point=0, cut_num=5):
    # tho = round(centerline.shape[0] * 0.06)
    centerline = centerline[cut_num:]

    direction_vector = get_direction_vector(centerline, ini_point, k=5)
    # plane_normal = get_plane_normal_vector(direction_vector)
    Areas = []
    counter = 0
    stable = 0

    # centerline = [centerline]
    # centerline = cupy.array(centerline)
    # centerline = np.array(centerline)


    for i in range(len(centerline)):

        center_point = centerline[i * 2]
        cross_section_points, area = extract_cross_section_new(binary_label, center_point, direction_vector)
        cross_section_volume = np.zeros_like(binary_label)
        for xi, yi, zi in cross_section_points:
            cross_section_volume[xi, yi, zi] = 1
        connect_region, num = extract_connected_region(cross_section_volume, center_point)
        area = connect_region.shape[0]
        Areas.append(area)
        if i == 0:
            final = connect_region
            continue
        if (Areas[i - 1] - Areas[i]) / Areas[i] >= 0.4:
            final = connect_region
            # centerline = centerline[i:]
            counter = i * 2
            break
        stable += 1
        if stable == 3:
            # centerline = centerline[i:]
            counter = i * 2
            break
        if (i + 1) * 2 >= len(centerline):
            counter = i * 2
            break

    return final, counter + cut_num


def get_neighbors(point):
    neighbors = []
    x, y, z = point
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i == 0 and j == 0 and k == 0:
                    continue
                xi, yi, zi = x + i, y + j, z + k
                neighbors.append([xi, yi, zi])
    return neighbors


def is_endves(centerline, endpoints):
    neighbors_begin = get_neighbors(centerline[0])

    # centerline = np.array(centerline)
    c = []
    for i in neighbors_begin:
        i = cupy.array(i)
        matches = np.all(endpoints == i, axis=1)
        if np.any(matches):
            c.append(centerline[1])
            return c
    neighbors_end = get_neighbors(centerline[1])
    for i in neighbors_end:
        i = cupy.array(i)
        matches = np.all(endpoints == i, axis=1)
        if np.any(matches):
            c.append(centerline[0])
            return c
    return centerline


def retain_duplicates(list1, list2):

    common_elements = set(list1) & set(list2)
    return list(common_elements)


def lst_toarray(datashape, list):
    data = np.zeros(datashape).astype(int)
    for xi, yi, zi in list:
        data[xi, yi, zi] = 1

    return data


def crop_vessel(patient, data_dir, predict=False):


    if predict:
        nifti_img = nib.load(rf'{data_dir}\{patient}\predaf.nii.gz')
    else:
        nifti_img = nib.load(rf'{data_dir}\{patient}\label.nii.gz')
    binary_label = nifti_img.get_fdata()
    binary_label = cupy.asarray(binary_label)

    binary_label[np.where(binary_label > 0.)] = 1

    data_dir = rf'{data_dir}\{patient}'
    centerline_list = [i for i in os.listdir(data_dir) if 'ctl_patch' in i]
    ves_edpoints = nib.load(rf'{data_dir}\degree2.nii.gz').get_fdata()
    ves_edpoints = cupy.asarray(ves_edpoints)

    ves_edpoints = np.argwhere(ves_edpoints == 1)
    point_index = 0
    # k = 3

    for i in tqdm(range(len(centerline_list))):
        all_ctl = []
        ctl_index = 0
        centerline_array = nib.load(os.path.join(data_dir, centerline_list[i])).get_fdata()
        print(centerline_list[i])
        # centerlines = np.argwhere(centerline_array == 1)
        centerline_array = cupy.asarray(centerline_array)
        ed_points = find_endpoints(centerline_array)
        start = is_endves(ed_points, ves_edpoints)
        for spoint in range(len(start)):
            if spoint == 0:
                centerline = sort_centerline(centerline_array, (start[spoint][0], start[spoint][1], start[spoint][2]))
            else:
                # print(type(centerline))
                # centerline = np.array(centerline)
                centerline = [row[:] for row in centerline[::-1]]

                a = binary_label[centerline[0][0], centerline[0][1], centerline[0][2]]
                print(a)
            # centerline = centerline[-ctl_index:]
            # centerline = np.array(centerline)

            cross_section_points, ctl_index = stable_areas(binary_label, centerline)
            centerline = centerline[ctl_index:]

            for j in cross_section_points:
                all_ctl.append(j)
        cross_section_volume = lst_toarray(binary_label.shape, all_ctl)
        # combined_nifti_img = nib.Nifti1Image(cross_section_volume, nifti_img.affine)
        # nib.save(combined_nifti_img, rf'{data_dir}\vessel_bcrop{i}.nii.gz')
        crop = binary_label - cross_section_volume
        # combined_nifti_img = nib.Nifti1Image(crop, nifti_img.affine)
        # nib.save(combined_nifti_img, rf'{data_dir}\vessel_crop{i}.nii.gz')
        croped, num = extract_connected_region(crop, centerline[len(centerline) // 2], crop=True)
        vessel = lst_toarray(binary_label.shape, list(croped))
        new_ctl = lst_toarray(binary_label.shape, centerline)

        combined_nifti_img = nib.Nifti1Image(vessel.get(), nifti_img.affine)
        nib.save(combined_nifti_img, rf'{data_dir}\vessel_patch{i}.nii.gz')
        combined_nifti_img = nib.Nifti1Image(new_ctl.get(), nifti_img.affine)
        nib.save(combined_nifti_img, rf'{data_dir}\newc_patch{i}.nii.gz')


    # direction_vector = get_direction_vector(centerline, point_index, k)
    # # plane_normal = get_plane_normal_vector(direction_vector)
    #

    # center_point = centerline[point_index]
    # cross_section_points = extract_cross_section_new(binary_label, center_point, direction_vector)

    # cross_section_volume = np.zeros_like(binary_label)
    # for xi, yi, zi in all_ctl:
    #     cross_section_volume[xi, yi, zi] = 1

    # combined_nifti_img = nib.Nifti1Image(cross_section_volume, nifti_img.affine)
    # nib.save(combined_nifti_img, r'D:\imgcas\10017784\final_combined_cross_sections.nii.gz')

    print("Combined cross sections saved as NIfTI file.")


