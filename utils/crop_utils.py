import numpy as np
import torch

def crop_volume(volume, patch_size=[96, 96, 96], overlap_size=[4, 4, 4]):
    """
    Crop a 3D volume into patches with specified size and overlap.
    Args:
        volume (np.ndarray): the 3D volume to be cropped, with shape [width, height, depth]
        patch_size (tuple or list): the size of patch, with format [patch_width, patch_height, patch_depth]
        overlap_size (tuple or list): the size of overlap between adjacent patches, with format [overlap_width, overlap_height, overlap_depth]
    Returns:
        np.ndarray: the cropped patches, with shape [num_patches, patch_width, patch_height, patch_depth]
    """
    depth, width, height = volume.shape[2:5]
    patch_width, patch_height, patch_depth = patch_size

    overlap_width, overlap_height, overlap_depth = overlap_size
    patches = []
    for z in range(0, depth - patch_depth + 1, patch_depth - overlap_depth):
        for y in range(0, height - patch_height + 1, patch_height - overlap_height):
            for x in range(0, width - patch_width + 1, patch_width - overlap_width):
                patch = volume[:,:, z:z+patch_depth ,x:x+patch_width, y:y+patch_height]
                patches.append(patch)
    #patches = np.asarray(patches)
    return patches

def merge_patches(patches, volume_size, overlap_size):
    """
    Merge the cropped patches into a complete 3D volume.
    Args:
        patches (np.ndarray): the cropped patches, with shape [num_patches, patch_width, patch_height, patch_depth]
        volume_size (tuple or list): the size of the complete volume, with format [width, height, depth]
        overlap_size (tuple or list): the size of overlap between adjacent patches, with format [overlap_width, overlap_height, overlap_depth]
    Returns:
        np.ndarray: the merged volume, with shape [width, height, depth]
    """
    depth, width, height = volume_size
    patch_depth,patch_width, patch_height = patches.shape[1:4]

    overlap_width, overlap_height, overlap_depth = overlap_size
    num_patches_x = (width - patch_width) // (patch_width - overlap_width) + 1
    num_patches_y = (height - patch_height) // (patch_height - overlap_height) + 1
    if patch_depth - overlap_depth == 0:
        num_patches_z = (depth - patch_depth) // (patch_depth - overlap_depth + 1) + 1
    else: num_patches_z = (depth - patch_depth) // (patch_depth - overlap_depth) + 1

    merged_volume = np.zeros(volume_size)
    data_volume = np.zeros(volume_size)
    idx = 0
    for z in range(num_patches_z):
        for y in range(num_patches_y):
            for x in range(num_patches_x):
                x_start = x * (patch_width - overlap_width)
                y_start = y * (patch_height - overlap_height)
                z_start = z * (patch_depth - overlap_depth)
                merged_volume[z_start:z_start+patch_depth, x_start:x_start+patch_width, y_start:y_start+patch_height] = patches[idx]
                data_volume = np.logical_or(merged_volume, data_volume)
                merged_volume = np.zeros(volume_size)
                idx += 1
    #merged_volume /= weight_volume
    return data_volume



def merge_patches_torch(patches, volume_size, overlap_size):
    """
    Merge the cropped patches into a complete 3D volume.
    Args:
        patches (np.ndarray): the cropped patches, with shape [num_patches, patch_width, patch_height, patch_depth]
        volume_size (tuple or list): the size of the complete volume, with format [width, height, depth]
        overlap_size (tuple or list): the size of overlap between adjacent patches, with format [overlap_width, overlap_height, overlap_depth]
    Returns:
        np.ndarray: the merged volume, with shape [width, height, depth]
    """
    depth, width, height = volume_size
    patch_depth,patch_width, patch_height = patches.shape[1:4]

    overlap_width, overlap_height, overlap_depth = overlap_size
    num_patches_x = (width - patch_width) // (patch_width - overlap_width) + 1
    num_patches_y = (height - patch_height) // (patch_height - overlap_height) + 1
    if patch_depth - overlap_depth == 0:
        num_patches_z = (depth - patch_depth) // (patch_depth - overlap_depth + 1) + 1
    else: num_patches_z = (depth - patch_depth) // (patch_depth - overlap_depth) + 1

    merged_volume = torch.zeros(volume_size, device='cuda')
    data_volume = torch.zeros(volume_size, device='cuda')
    idx = 0
    for z in range(num_patches_z):
        for y in range(num_patches_y):
            for x in range(num_patches_x):
                x_start = x * (patch_width - overlap_width)
                y_start = y * (patch_height - overlap_height)
                z_start = z * (patch_depth - overlap_depth)
                merged_volume[z_start:z_start+patch_depth, x_start:x_start+patch_width, y_start:y_start+patch_height] = patches[idx]
                data_volume = torch.logical_or(merged_volume, data_volume)
                merged_volume = torch.zeros(volume_size, device='cuda')
                idx += 1
    #merged_volume /= weight_volume

    return data_volume


def merge_patches_s(patches, volume_size, overlap_size):

    depth, height, width = volume_size
    patch_depth, patch_height, patch_width = patches.shape[1:]
    overlap_height, overlap_width, overlap_depth = overlap_size
    if patch_depth - overlap_depth == 0:
        num_patches_z = (depth - patch_depth) // (patch_depth - overlap_depth + 1) + 1
    else:
        num_patches_z = (depth - patch_depth) // (patch_depth - overlap_depth) + 1
    num_patches_x = (height - patch_height) // (patch_height - overlap_height) + 1
    num_patches_y = (width - patch_width) // (patch_width - overlap_width) + 1

    print('merge:', num_patches_z, num_patches_x, num_patches_y)
    merged_volume = torch.zeros(volume_size, device='cuda')
    weight_volume = torch.zeros(volume_size, device='cuda')
    idx = 0
    for z in range(num_patches_z):
        for x in range(num_patches_x):
            for y in range(num_patches_y):
                z_start = z * (patch_depth - overlap_depth)
                x_start = x * (patch_height - overlap_height)
                y_start = y * (patch_width - overlap_width)

                merged_volume[z_start:z_start + patch_depth, x_start:x_start + patch_height,
                y_start:y_start + patch_width] += patches[idx]
                weight_volume[z_start:z_start + patch_depth, x_start:x_start + patch_height,
                y_start:y_start + patch_width] += 1
                idx += 1
    merged_volume /= (weight_volume + 1e-10)  # 肯定有小数, 有nan 值
    # merged_volume=np.divide(merged_volume,weight_volume)
    # merged_volume=MinMaxScale(merged_volume)
    return merged_volume


def crop_volume_reverse_uniform(volume, patch_size=[96, 96, 96], overlap_size=[4, 4, 4]):
    """
    Reverse crop a 3D volume into patches ensuring complete coverage with overlap.
    Ensures all patches are of uniform size by adjusting positions and padding if necessary.

    Args:
        volume (np.ndarray): the 3D volume to be cropped, with shape [depth, width, height]
        patch_size (tuple or list): the size of patch, with format [patch_depth, patch_width, patch_height]
        overlap_size (tuple or list): the size of overlap between adjacent patches, with format [overlap_depth, overlap_width, overlap_height]

    Returns:
        np.ndarray: the cropped patches, with shape [num_patches, patch_depth, patch_width, patch_height]
    """
    depth, width, height = volume.shape[2:5]
    patch_depth, patch_width, patch_height = patch_size
    overlap_depth, overlap_width, overlap_height = overlap_size

    # Compute the step sizes
    step_depth = patch_depth - overlap_depth
    step_width = patch_width - overlap_width
    step_height = patch_height - overlap_height

    # Calculate the number of patches needed in each dimension
    num_patches_depth = (depth - overlap_depth) // step_depth + 1
    num_patches_width = (width - overlap_width) // step_width + 1
    num_patches_height = (height - overlap_height) // step_height + 1
    # print('x:', num_patches_width, "y:", num_patches_height, "z:", num_patches_depth)
    num = num_patches_depth*num_patches_width*num_patches_height

    patches = torch.zeros((num,1,1,patch_depth, patch_width, patch_height), device='cuda')
    index = 0

    # Iterate through the computed positions, including handling the edges
    for i in range(num_patches_depth):
        for j in range(num_patches_width):
            for k in range(num_patches_height):
                # Calculate starting indices
                z = i * step_depth
                x = j * step_width
                y = k * step_height
                # print('x:',x,"y:",y,"z:",z)

                # Adjust starting index to ensure full coverage of the volume
                if z + patch_depth > depth:
                    z = depth - patch_depth
                if y + patch_height > height:
                    # print('y',y, y + patch_height, height)
                    y = height - patch_height
                    # print('y',y, y + patch_height, height)
                if x + patch_width > width:
                    # print('x',x, x + patch_width, width)
                    x = width - patch_width
                    # print('x',x, x + patch_width, width)



                # Extract the patch
                patch = volume[:,:,z:z + patch_depth, x:x + patch_width, y:y + patch_height]
                patches[index] = patch
                index+=1
    # print(len(patches))

    return patches










def assemble_patches_single_np(all, nums, patch, original_shape, patch_size=[96, 96, 96], overlap_size=[4, 4, 4]):

    width, height, depth = original_shape
    patch_depth, patch_width, patch_height = patch_size
    overlap_depth, overlap_width, overlap_height = overlap_size

    # Initialize the volume and a weight array to account for overlapping regions

    # merged_volume = torch.zeros((depth, width, height), device='cuda')
    data_volume = all
    # Compute the step sizes
    step_depth = patch_depth - overlap_depth
    step_width = patch_width - overlap_width
    step_height = patch_height - overlap_height

    # Calculate the number of patches needed in each dimension
    num_patches_depth = (depth - overlap_depth) // step_depth + 1
    num_patches_width = (width - overlap_width) // step_width + 1
    num_patches_height = (height - overlap_height) // step_height + 1

    # Iterate through the computed positions and place patches
    patch_idx = 0
    for i in range(num_patches_depth):
        for j in range(num_patches_width):
            for k in range(num_patches_height):
                # Calculate starting indices
                z = i * step_depth
                x = j * step_width  # Corrected
                y = k * step_height  # Corrected

                # Adjust starting index to ensure full coverage of the volume
                if z + patch_depth > depth:
                    z = depth - patch_depth
                if x + patch_width > width:
                    x = width - patch_width
                if y + patch_height > height:
                    y = height - patch_height

                # Place the patch in the volume
                if patch_idx == nums:
                    # merged_volume[z:z + patch_depth, x:x + patch_width, y:y + patch_height] = patch
                    a = data_volume[z:z + patch_depth, x:x + patch_width, y:y + patch_height]
                    data_volume[z:z + patch_depth, x:x + patch_width, y:y + patch_height] = np.logical_or(patch, a)
                    return data_volume

                patch_idx += 1

    # Normalize by the weight to account for overlapping areas
    # volume /= torch.maximum(weight, 1)

    return data_volume

def assemble_patches_single(all, nums, patch, original_shape, patch_size=[96, 96, 96], overlap_size=[4, 4, 4]):

    width, height, depth = original_shape
    patch_depth, patch_width, patch_height = patch_size
    overlap_depth, overlap_width, overlap_height = overlap_size

    # Initialize the volume and a weight array to account for overlapping regions

    # merged_volume = torch.zeros((depth, width, height), device='cuda')

    data_volume = all
    # Compute the step sizes
    step_depth = patch_depth - overlap_depth
    step_width = patch_width - overlap_width
    step_height = patch_height - overlap_height

    # Calculate the number of patches needed in each dimension
    num_patches_depth = (depth - overlap_depth) // step_depth + 1
    num_patches_width = (width - overlap_width) // step_width + 1
    num_patches_height = (height - overlap_height) // step_height + 1

    # Iterate through the computed positions and place patches
    patch_idx = 0
    for i in range(num_patches_depth):
        for j in range(num_patches_width):
            for k in range(num_patches_height):
                # Calculate starting indices
                z = i * step_depth
                x = j * step_width  # Corrected
                y = k * step_height  # Corrected

                # Adjust starting index to ensure full coverage of the volume
                if z + patch_depth > depth:
                    z = depth - patch_depth
                if x + patch_width > width:
                    x = width - patch_width
                if y + patch_height > height:
                    y = height - patch_height

                # Place the patch in the volume
                if patch_idx == nums:
                    data_volume[z:z + patch_depth, x:x + patch_width, y:y + patch_height] = patch
                    # data_volume = torch.logical_or(data_volume, merged_volume)
                    return data_volume

                patch_idx += 1

    return data_volume


def assemble_patches_singleweight(nums, original_shape, patch_size=[96, 96, 96], overlap_size=[4, 4, 4]):
    width, height, depth = original_shape
    patch_depth, patch_width, patch_height = patch_size
    overlap_depth, overlap_width, overlap_height = overlap_size

    # Initialize the volume and a weight array to account for overlapping regions

    # merged_volume = torch.zeros((depth, width, height), device='cuda')
    # weight_volume = weights

    # data_volume = torch.zeros((depth, width, height), device='cuda')
    # Compute the step sizes
    step_depth = patch_depth - overlap_depth
    step_width = patch_width - overlap_width
    step_height = patch_height - overlap_height

    # Calculate the number of patches needed in each dimension
    num_patches_depth = (depth - overlap_depth) // step_depth + 1
    num_patches_width = (width - overlap_width) // step_width + 1
    num_patches_height = (height - overlap_height) // step_height + 1

    # Iterate through the computed positions and place patches
    patch_idx = 0
    for i in range(num_patches_depth):
        for j in range(num_patches_width):
            for k in range(num_patches_height):
                # Calculate starting indices
                z = i * step_depth
                x = j * step_width  # Corrected
                y = k * step_height  # Corrected

                # Adjust starting index to ensure full coverage of the volume
                if z + patch_depth > depth:
                    z = depth - patch_depth
                if x + patch_width > width:
                    x = width - patch_width
                if y + patch_height > height:
                    y = height - patch_height

                # Place the patch in the volume
                if patch_idx == nums:
                    # weight_volume[z:z + patch_depth, x:x + patch_width, y:y + patch_height] += 1
                    # merged_volume[z:z + patch_depth, x:x + patch_width, y:y + patch_height] = patch
                    # data_volume[z:z + patch_depth, x:x + patch_width, y:y + patch_height] = 1


                    return z,z + patch_depth, x,x + patch_width, y,y + patch_height

                patch_idx += 1

    return z,z + patch_depth, x,x + patch_width, y,y + patch_height



def assemble_patches(patches, original_shape, patch_size=[96, 96, 96], overlap_size=[4, 4, 4]):
    """
    Assembles patches back into the original 3D volume, considering overlaps.

    Args:
        patches (np.ndarray): the patches to be assembled, with shape [num_patches, patch_depth, patch_width, patch_height]
        original_shape (tuple): the shape of the original 3D volume, with format [depth, width, height]
        patch_size (tuple or list): the size of patch, with format [patch_depth, patch_width, patch_height]
        overlap_size (tuple or list): the size of overlap between adjacent patches, with format [overlap_depth, overlap_width, overlap_height]

    Returns:
        np.ndarray: the reassembled 3D volume
    """
    width, height, depth = original_shape
    patch_depth, patch_width, patch_height = patch_size
    overlap_depth, overlap_width, overlap_height = overlap_size

    # Initialize the volume and a weight array to account for overlapping regions

    merged_volume = torch.zeros((depth, width, height), device='cuda')
    data_volume = torch.zeros((depth, width, height), device='cuda')
    weight_volume = torch.zeros((depth, width, height), device='cuda')
    # Compute the step sizes
    step_depth = patch_depth - overlap_depth
    step_width = patch_width - overlap_width
    step_height = patch_height - overlap_height

    # Calculate the number of patches needed in each dimension
    num_patches_depth = (depth - overlap_depth) // step_depth + 1
    num_patches_width = (width - overlap_width) // step_width + 1
    num_patches_height = (height - overlap_height) // step_height + 1

    # Iterate through the computed positions and place patches
    patch_idx = 0
    for i in range(num_patches_depth):
        for j in range(num_patches_width):
            for k in range(num_patches_height):
                # Calculate starting indices
                z = i * step_depth
                x = j * step_width  # Corrected
                y = k * step_height  # Corrected

                # Adjust starting index to ensure full coverage of the volume
                if z + patch_depth > depth:
                    z = depth - patch_depth
                if x + patch_width > width:
                    x = width - patch_width
                if y + patch_height > height:
                    y = height - patch_height

                # Place the patch in the volume
                data_volume[z:z + patch_depth, x:x + patch_width, y:y + patch_height] = patches[patch_idx]
                # weight_volume[z:z + patch_depth, x:x + patch_width, y:y + patch_height] += 1
                # data_volume = torch.logical_or(merged_volume, data_volume)
                # merged_volume = torch.zeros((depth, width, height), device='cuda')

                patch_idx += 1
    # data_volume /= (weight_volume + 1e-10)

    # Normalize by the weight to account for overlapping areas
    # volume /= torch.maximum(weight, 1)

    return data_volume



def assemble_patchesnp(patches, original_shape, patch_size=[96, 96, 96], overlap_size=[4, 4, 4]):
    """
    Assembles patches back into the original 3D volume, considering overlaps.

    Args:
        patches (np.ndarray): the patches to be assembled, with shape [num_patches, patch_depth, patch_width, patch_height]
        original_shape (tuple): the shape of the original 3D volume, with format [depth, width, height]
        patch_size (tuple or list): the size of patch, with format [patch_depth, patch_width, patch_height]
        overlap_size (tuple or list): the size of overlap between adjacent patches, with format [overlap_depth, overlap_width, overlap_height]

    Returns:
        np.ndarray: the reassembled 3D volume
    """
    width, height, depth = original_shape
    patch_depth, patch_width, patch_height = patch_size
    overlap_depth, overlap_width, overlap_height = overlap_size

    # Initialize the volume and a weight array to account for overlapping regions

    weight_volume = np.zeros((depth, width, height))
    data_volume = np.zeros((depth, width, height))
    # Compute the step sizes
    step_depth = patch_depth - overlap_depth
    step_width = patch_width - overlap_width
    step_height = patch_height - overlap_height

    # Calculate the number of patches needed in each dimension
    num_patches_depth = (depth - overlap_depth) // step_depth + 1
    num_patches_width = (width - overlap_width) // step_width + 1
    num_patches_height = (height - overlap_height) // step_height + 1

    # Iterate through the computed positions and place patches
    patch_idx = 0
    for i in range(num_patches_depth):
        for j in range(num_patches_width):
            for k in range(num_patches_height):
                # Calculate starting indices
                z = i * step_depth
                x = j * step_width  # Corrected
                y = k * step_height  # Corrected

                # Adjust starting index to ensure full coverage of the volume
                if z + patch_depth > depth:
                    z = depth - patch_depth
                if x + patch_width > width:
                    x = width - patch_width
                if y + patch_height > height:
                    y = height - patch_height

                # Place the patch in the volume
                data_volume[z:z + patch_depth, x:x + patch_width, y:y + patch_height] += patches[patch_idx]
                weight_volume[z:z + patch_depth, x:x + patch_width, y:y + patch_height] += 1
                # data_volume = np.logical_or(merged_volume, data_volume)
                # merged_volume = np.zeros((depth, width, height))

                patch_idx += 1
    data_volume /= (weight_volume + 1e-10)

    # Normalize by the weight to account for overlapping areas
    # volume /= torch.maximum(weight, 1)

    return data_volume


