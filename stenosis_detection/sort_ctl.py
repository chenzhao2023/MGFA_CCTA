import numpy
import nibabel as nib
from scipy.ndimage import label
import cupy
import cupy as np


def find_endpoints(centerline):

    endpoints = []
    centerline = cupy.array(centerline)
    centerline_points = np.argwhere(centerline == 1)
    for point in centerline_points:
        x, y, z = point

        neighbors = centerline[x-1:x+2, y-1:y+2, z-1:z+2]
        num_neighbors = np.sum(neighbors) - 1
        if num_neighbors == 1:
            endpoints.append((x, y, z))
    return endpoints

def sort_centerline(centerline, start_point):

    points = []
    points.append(start_point)
    current_point = start_point
    # centerline_points = np.argwhere(centerline == 1)
    while True:
        x, y, z = current_point
        # centerline = centerline.get()

        x, y, z = x.get(), y.get(), z.get()

        centerline[x, y, z] = 0

        neighbors = centerline[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]


        # num_neighbors = np.sum(neighbors) - 1
        num_neighbors = np.sum(neighbors)
        neighbors = np.array(neighbors)
        if num_neighbors == 1:
            index = np.argwhere(neighbors == 1)[0]

            offset = (index[0] - 1, index[1] - 1, index[2] - 1)
            # offset = offset.get()
            # x, y, z = np.array(x), y.get(), z.get()
            x, y, z = np.array(x), np.array(y), np.array(z)
            offset = np.array(offset)  # 确保 offset 是 NumPy 数组
            current_point = (x + offset[0], y + offset[1], z + offset[2])
            points.append(current_point)

        if num_neighbors == 0:
            break
    return points
    # return sorted_points

