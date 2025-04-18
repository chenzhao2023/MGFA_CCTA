import nibabel as nib
import numpy as np
import os
from tqdm import tqdm

def dual_image_cropper(mask_path, image_path,
                       output_mask_path=None, output_image_path=None,
                       threshold=0.95, target_size=(384, 384, 256)):

    try:

        mask_img = nib.load(mask_path)
        image_img = nib.load(image_path)


        if mask_img.shape[:3] != image_img.shape[:3]:
            raise ValueError(f"{mask_img.shape} vs {image_img.shape}")

        mask_data = mask_img.get_fdata(dtype=np.float32)
        image_data = image_img.get_fdata(dtype=np.float32)
        original_shape = mask_img.shape[:3]


        crop_ranges = []
        for orig_dim, target_dim in zip(original_shape, target_size):
            if orig_dim > target_dim:
                start = (orig_dim - target_dim) // 2
                end = start + target_dim
            else:
                start, end = 0, orig_dim
            crop_ranges.append((start, end))

        cropped_mask = mask_data[
                       crop_ranges[0][0]:crop_ranges[0][1],
                       crop_ranges[1][0]:crop_ranges[1][1],
                       crop_ranges[2][0]:crop_ranges[2][1]
                       ]

        cropped_image = image_data[
                        crop_ranges[0][0]:crop_ranges[0][1],
                        crop_ranges[1][0]:crop_ranges[1][1],
                        crop_ranges[2][0]:crop_ranges[2][1]
                        ]


        original_foreground = mask_data.sum()
        if original_foreground == 0:
            return False, 0.0, None, None
        preserved_foreground = cropped_mask.sum()
        ratio = preserved_foreground / original_foreground


        def update_affine(original_affine):
            new_affine = original_affine.copy()
            start_indices = [r[0] for r in crop_ranges]
            start_phys = nib.affines.apply_affine(original_affine, start_indices)
            new_affine[:3, 3] = start_phys
            return new_affine

        mask_affine = update_affine(mask_img.affine)
        image_affine = update_affine(image_img.affine)

        cropped_mask_img = nib.Nifti1Image(cropped_mask, mask_affine)
        cropped_image_img = nib.Nifti1Image(cropped_image, image_affine)


        save_flag = ratio >= threshold
        if save_flag:
            if output_mask_path is not None:
                nib.save(cropped_mask_img, output_mask_path)
            if output_image_path is not None:
                nib.save(cropped_image_img, output_image_path)

        return save_flag, ratio, cropped_mask.shape, cropped_image.shape

    except Exception as e:
        print(f"error: {str(e)}")
        return False, 0.0, None, None




if __name__ == '__main__':
    datadir = r''
    for i in tqdm(os.listdir(datadir)):




        mask_file = rf""
        image_file = rf""
        output_mask = output_file = rf""
        output_image = rf""

        result = dual_image_cropper(
            mask_path=mask_file,
            image_path=image_file,
            output_mask_path=output_mask,
            output_image_path=output_image,
            threshold=0.95
        )

        print(f"""
        result:
        state: {result[0]}
        {result[1]:.2%}
        {result[2]}
        {result[3]}
        """)