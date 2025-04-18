# import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import nibabel as nib
from CSA import extract_cross_section_new, is_endves
from CSA import lst_toarray
from scipy.ndimage import label
from sort_ctl import find_endpoints, sort_centerline
import pandas as pd
import cupy as np
import cupy




def calculate_pixel_spacing(direction_vector, voxel_spacing):

    if np.allclose(direction_vector, [1.0, 0.0, 0.0]):
        basis1 = np.array([0.0, 1.0, 0.0])
    else:
        basis1 = np.cross(direction_vector, [1.0, 0.0, 0.0])
        basis1 /= np.linalg.norm(basis1)

    basis2 = np.cross(direction_vector, basis1)
    basis2 /= np.linalg.norm(basis2)


    pixel_spacing_1 = np.sqrt(np.sum((basis1 * voxel_spacing) ** 2))
    pixel_spacing_2 = np.sqrt(np.sum((basis2 * voxel_spacing) ** 2))

    return pixel_spacing_1, pixel_spacing_2


def compute_true_areas(voxel_spacing, vector, areas):
    p1, p2 = calculate_pixel_spacing(vector, voxel_spacing)

    a3 = areas * p1 * p2
    return a3


def compute_curvature_vectors(centerline):
    curvature_vectors = np.zeros_like(centerline)
    for i in range(1, len(centerline) - 1):
        p0 = centerline[i - 1]
        p1 = centerline[i]
        p2 = centerline[i + 1]

        v1 = p1 - p0
        v2 = p2 - p1
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)

        if d1 == 0 or d2 == 0:
            continue

        v1_normalized = v1 / d1
        v2_normalized = v2 / d2

        cos_theta = np.clip(np.dot(v1_normalized, v2_normalized), -1.0, 1.0)
        theta = np.arccos(cos_theta)

        curvature = 2 * theta / (d1 + d2)

        curvature_vector = np.cross(v1_normalized, v2_normalized)
        curvature_vector_norm = np.linalg.norm(curvature_vector)
        if curvature_vector_norm != 0:
            curvature_vector /= curvature_vector_norm

        curvature_vectors[i] = curvature * curvature_vector

    return curvature_vectors


def extract_connected_region(element_indices, center, shape):

    labels = np.zeros(shape, dtype=int)
    labels[tuple(element_indices.T)] = 1

    labeled_array, num_features = label(labels, structure=np.ones((3, 3, 3)))

    x, y, z = center
    center_label = labeled_array[x, y, z]

    connected_region = np.argwhere(labeled_array == center_label)

    return connected_region, num_features


def calculate_angle_between_vectors(vector1, vector2):
    dot_product = np.dot(vector1, vector2)

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    cos_theta = dot_product / (norm1 * norm2)

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta)

    angle_deg = np.degrees(angle_rad)

    return angle_deg


def get_direction_vector(centerline, index, k=3, direction=1):

    start_index = max(0, index - k)
    end_index = min(len(centerline) - 1, index + k)
    if direction == 2:
        start_index = index
    elif direction == 0:
        end_index = index
    start_point = centerline[start_index]
    end_point = centerline[end_index]
    end_point= np.array(end_point)
    start_point = np.array(start_point)
    direction_vector = end_point - start_point
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # 归一化
    return direction_vector

def record_stens(min, max, point, data):
    percent = (1-(min/max))*100
    percent = percent
    print(min, max, percent)
    st = 0
    # print(f"point: {point}, type: {type(point)}, shape: {point.shape}")
    point = np.array(point)
    point= point.get()

    # data = data.get()

    if 24 >= percent >= 1:
        data[point[0], point[1], point[2]] = 1
        st = 0
    elif 49 >= percent >= 25:
        data[point[0], point[1]-5:point[1]+5, point[2]] = 1
        data[point[0], point[1], point[2] - 5:point[2] + 5] = 1
        data[point[0] - 5:point[0] + 5, point[1], point[2]] = 1
        st = 1
    elif 69 >= percent >= 50:
        data[point[0], point[1] - 20:point[1] + 20, point[2]] = 1
        data[point[0], point[1], point[2] - 20:point[2] + 20] = 1
        data[point[0] - 20:point[0] + 20, point[1], point[2]] = 1
        st = 2
    elif 100 >= percent >= 70:
        data[point[0], point[1] - 50:point[1] + 50, point[2]] = 1
        data[point[0], point[1], point[2] - 50:point[2] + 50] = 1
        data[point[0] - 50:point[0] + 50, point[1], point[2]] = 1
        st = 3
    return data, percent, st

def compute_one_derivative(areas):
    n = len(areas)
    one_derivative = np.zeros(n)
    for i in range(0, n-1):
        if areas[i+1] - areas[i] > 0:
            one_derivative[i] = 1
        else:
            one_derivative[i] = -1
    return one_derivative

def compute_second_derivative(areas):
    n = len(areas)
    second_derivative = np.zeros(n)
    for i in range(0, n-2):
        second_derivative[i] = areas[i+1] - areas[i]
    return second_derivative

def compute_thrid_derivative(areas):
    n = len(areas)
    thrid_derivative = np.zeros(n)
    for i in range(0, n-3):
        thrid_derivative[i] = areas[i+1] - areas[i]
    return thrid_derivative


def central_difference_second_derivative(areas, h=1.0):

    n = len(areas)
    second_derivative = np.zeros(n)
    for i in range(1, n - 1):
        second_derivative[i] = (areas[i + 1] - 2 * areas[i] + areas[i - 1]) / (h ** 2)
    return second_derivative


def central_difference_third_derivative(areas, h=1.0):

    n = len(areas)
    third_derivative = np.zeros(n)
    for i in range(2, n - 2):
        third_derivative[i] = (-areas[i + 2] + 2 * areas[i + 1] - 2 * areas[i - 1] + areas[i - 2]) / (2 * h ** 3)
    return third_derivative


def compute_third_derivative(areas):
    n = len(areas)
    third_derivative = np.zeros(n)
    for i in range(3, n):
        third_derivative[i] = (areas[i] - 3 * areas[i - 1] + 3 * areas[i - 2] - areas[i - 3])
    return third_derivative


def find_top_k_min_indices(array, k):
    return np.argsort(array)[:k]


def find_top_k_max_indices(array, k):
    return np.argsort(array)[-k:]


#
def detect_stenosis(areas, threshold=0):
    one_der = compute_one_derivative(areas)
    second_der = compute_second_derivative(one_der)
    # second_der = compute_third_derivative(second_der)


    # second_derivative = central_difference_second_derivative(areas)
    stenosis_points = np.where(second_der == -threshold)[0]
    max_points = np.where(second_der == threshold)[0]
    # print(len(stenosis_points), len(max_points))

    # stenosis_points = stenosis_points.astype(int) - 1

    min_area_index_local = np.argmin(cupy.array([areas[i.item()] for i in stenosis_points]))
    stenosis_points = stenosis_points[min_area_index_local]
    max_area_index_local = np.argmax(cupy.array([areas[i.item()] for i in max_points]))
    max_point = max_points[max_area_index_local]

    return stenosis_points, max_point, second_der


# def save_stenosis(label_shape, array, patient_dir, aff):
#     axis = np.zeros(label_shape).astype(int)
#     for x in array:
#         axis[x[0], x[1], x[2]] = 1
#     combined_nifti_img = nib.Nifti1Image(axis, aff)
#     nib.save(combined_nifti_img, r'..\{}\stenosis_all.nii.gz'.format(patient_dir))
#     print('save')



def get_stenosis(data_dir, patient):

    st_area_xlsx = {
        'position': [],
        'stenosis': [],
        'vessel_num': [],
        'percent': []
    }

    df_area = pd.DataFrame(st_area_xlsx)

    root_dir = rf'{data_dir}\{patient}'

    ves_edpoints = nib.load(rf'{data_dir}\{patient}\degree2.nii.gz').get_fdata()
    ves_edpoints = cupy.asarray(ves_edpoints)
    # stenosis_data = np.zeros_like(ves_edpoints)
    stenosis_area = np.zeros_like(ves_edpoints)
    # pixel_area = np.zeros_like(ves_edpoints)
    # stenosis_data_2 = np.zeros_like(ves_edpoints)
    ves_edpoints = np.argwhere(ves_edpoints == 1)


    vessel_list = [i for i in os.listdir(root_dir) if r'vessel_patch' in i]
    all_cross = []
    # all_cross = np.array(all_cross)
    # all_stenosis_points = []
    Drop = []

    for index, d in enumerate(vessel_list):


        vessel_dir = os.path.join(root_dir, d)
        ctl_dir = os.path.join(root_dir, d.replace('vessel_patch', 'newc_patch'))

        centerline_array = nib.load(ctl_dir)
        # header = centerline_array.header
        aff = centerline_array.affine
        voxel_spacing = np.abs(np.diag(aff)[:3])
        centerline_array = nib.load(ctl_dir).get_fdata()
        # centerline_array = np.array(centerline_array)
        labels = nib.load(vessel_dir).get_fdata()
        labels = cupy.asarray(labels)
        # labels_num = np.sum(labels)
        if index == 0:
            all_labels = labels

        else:
            all_labels = np.logical_or(labels, all_labels)
        ed_points = find_endpoints(centerline_array)
        if len(ed_points)==0:
            continue

        centerline = sort_centerline(centerline_array, (ed_points[0][0], ed_points[0][1], ed_points[0][2]))
        # centerline = np.array(centerline)
        # start = is_endves(ed_points, ves_edpoints)

        # aff = nib.load(os.path.join(dir, 'vessel_block.nii.gz')).affine
        # centerline = np.argwhere(centerline_array == 1)
        Areas_all = []
        True_area = []
        # True_diameter = []
        for i in range(len(centerline)):
            Areas = []
            Cross = []
            Vector = []

            for s in [8, 10, 15]:

                vector = get_direction_vector(centerline, i, s, direction=1)
                pixels, areas = extract_cross_section_new(labels, centerline[i], vector)


                Vector.append(vector)
                Cross.append(pixels)
                Areas.append(areas)

            min_index = np.argmin(cupy.array(Areas))

            if i % 3 == 0:
                min_index = min_index.get()
                for v in Cross[min_index]:
                    all_cross.append(v)
                min_index = np.argmin(cupy.array(Areas))

            ture_area = compute_true_areas(voxel_spacing, Vector[min_index.item()], Areas[min_index.item()])
            # true_diameter = np.sqrt(ture_area / np.pi)*2
            # True_diameter.append(true_diameter)
            True_area.append(ture_area)
            # if 1.8 > ture_area > 0:
            #     for v in Cross[min_index]:
            #         Drop.append(v)
                # continue

            Areas_all.append(Areas[min_index.item()])




            # print(f"{d}:   {i}:Cross-sectional areas: {np.min(Areas)}, Ture diameter: {true_diameter}")
            # labels = crop_label(labels, Cross[min_index])
        if len(Areas_all) < 30:
            print(r'vessel{}'.format(index) + r'  too short.......')
        else:
            # Areas_all = Areas_all[5:-5]
            # True_diameter = True_diameter[5:-5]
            # True_area = True_area[5:-5]
            # stenosis_points, maxpoint, second_derivative = detect_stenosis(Areas_all, 0)

            tho = 2
            # stenosis_points, maxpoint, second_derivative = detect_stenosis(True_diameter[5:-5], tho)
            # stenosis_points = stenosis_points + 5
            # maxpoint = maxpoint + 5

            # stenosis_data, percent, st = record_stens(True_diameter[stenosis_points], True_diameter[maxpoint],
            #                              centerline[stenosis_points], stenosis_data)
            # record_xlsx = pd.DataFrame(
            #     {'position': [centerline[stenosis_points]], 'stenosis': [st], 'vessel_num': [index],
            #      'percent': [percent]})
            # df = pd.concat([df, record_xlsx], ignore_index=True)


            # True_diameter = [round(num, 2) for num in True_diameter]
            # stenosis_points, maxpoint, second_derivative = detect_stenosis(True_diameter[5:-5], tho)
            # stenosis_points = stenosis_points + 5
            # maxpoint = maxpoint + 5
            # stenosis_data_2, _, st = record_stens(True_diameter[stenosis_points], True_diameter[maxpoint],
            #                              centerline[stenosis_points], stenosis_data_2)


            stenosis_points, maxpoint, second_derivative = detect_stenosis(True_area[5:-5], tho)
            stenosis_points = stenosis_points + 5
            maxpoint = maxpoint + 5
            stenosis_area, percent_area, st = record_stens(True_area[stenosis_points.item()], True_area[maxpoint.item()], centerline[stenosis_points.item()],stenosis_area)
            # centerline = centerline.get()
            # st = st.get()
            # index = index.get()
            # percent_area = percent_area.get()
            stenosis_points = stenosis_points.get()
            percent_area = percent_area.get()
            record_area_xlsx = pd.DataFrame(
                {'position': [centerline[stenosis_points]], 'stenosis': [st], 'vessel_num': [index],
                 'percent': [percent_area]})
            df_area = pd.concat([df_area, record_area_xlsx], ignore_index=True)




            # stenosis_points, maxpoint, second_derivative = detect_stenosis(Areas_all[5:-5], tho)
            # stenosis_points = stenosis_points + 5
            # maxpoint = maxpoint + 5
            # pixel_area, _, st = record_stens(Areas_all[stenosis_points], Areas_all[maxpoint], centerline[stenosis_points],
            #                              pixel_area)
            # for st in stenosis_points:


            # all_stenosis_points.append(centerline[stenosis_points])

        # if len(start) == 1:
        #     print(labels_num, np.sum(Areas_all), 'end')
        # else:
        #     print(labels_num, np.sum(Areas_all))

    # save_stenosis(all_labels.shape, all_stenosis_points, patient, aff)


    # stenosis_data = stenosis_data.astype(int)
    # combined_nifti_img = nib.Nifti1Image(stenosis_data, aff)
    # nib.save(combined_nifti_img, rf'{data_dir}\{patient}\stenosis_data.nii.gz')
    #
    # stenosis_data_2 = stenosis_data_2.astype(int)
    # combined_nifti_img = nib.Nifti1Image(stenosis_data_2, aff)
    # nib.save(combined_nifti_img, rf'{data_dir}\{patient}\stenosis_data_2.nii.gz')
    #
    stenosis_area = stenosis_area.astype(int)
    combined_nifti_img = nib.Nifti1Image(stenosis_area.get(), aff)
    nib.save(combined_nifti_img, rf'{data_dir}\{patient}\stenosis_area.nii.gz')
    #
    # piexl_area = pixel_area.astype(int)
    # combined_nifti_img = nib.Nifti1Image(piexl_area, aff)
    # nib.save(combined_nifti_img, rf'{data_dir}\{patient}\piexl_area.nii.gz')
    #
    # Drop = lst_toarray(all_labels.shape, Drop)
    # Drop = Drop.astype(int)
    # combined_nifti_img = nib.Nifti1Image(Drop, aff)
    # nib.save(combined_nifti_img, rf'{data_dir}\{patient}\vessel_drop.nii.gz')
    #
    all_cross = lst_toarray(all_labels.shape, all_cross)
    all_cross = all_cross.astype(int)
    combined_nifti_img = nib.Nifti1Image(all_cross.get(), aff)
    nib.save(combined_nifti_img, rf'{data_dir}\{patient}\vessel_cross_all.nii.gz')
    #
    # all_labels = all_labels.astype(int)
    # combined_nifti_img = nib.Nifti1Image(all_labels, aff)
    # nib.save(combined_nifti_img, rf'{data_dir}\{patient}\vessel_all.nii.gz')
    #
    # df.to_excel(rf'{data_dir}\{patient}\st_data.xlsx', index=False)
    df_area.to_excel(rf'{data_dir}\{patient}\st_area_data.xlsx', index=False)
    # cross = cross.astype(float)
    # combined_nifti_img = nib.Nifti1Image(cross, aff)
    # number = d.split('.')[0]
    # number = number.split('patch')[1]
    # nib.save(combined_nifti_img, r'D:\imgcas\{}\vessel_cross{}.nii.gz'.format('10018985', number))




