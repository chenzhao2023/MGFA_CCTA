import pyvista as pv
import numpy as np
import pandas as pd
import re



def parse_position_string(position_str):

    if 'array' in position_str:
        cleaned_str = re.sub(r'array\(([^,]+),?\s*dtype=int64\)', r'\1', position_str)
        cleaned_str = re.sub(r'[^\d,]', '', cleaned_str)
        return eval(f"[{cleaned_str}]")
    else:
        cleaned_str = re.sub(r'\s+', ', ', position_str.strip())
        return eval(cleaned_str)



def assign_color(percent, brightness_factor=0.7):

    if percent <= 25:
        base_color = [0, 255, 0]
    elif percent <= 50:
        base_color = [0, 0, 255]
    elif percent <= 70:
        base_color = [255, 255, 0]
    else:
        base_color = [255, 0, 0]


    brightened_color = [
        int((1 - brightness_factor) * c + brightness_factor * 255) for c in base_color
    ]
    return brightened_color


def load_vessel_image(nii_file):
    import nibabel as nib
    img = nib.load(nii_file)
    data = img.get_fdata()
    return data


def create_stenosis_actor(stenosis_data):
    points = []
    colors = []

    for _, row in stenosis_data.iterrows():
        position = parse_position_string(row['position'])
        percent = row['percent']
        points.append(position)
        colors.append(assign_color(percent, brightness_factor=0.1))

    points = np.array(points)
    colors = np.array(colors)

    stenosis_cloud = pv.PolyData(points)
    stenosis_cloud['Colors'] = colors

    return stenosis_cloud


def visualize_vessel_comparison(vessel_data, stenosis_data):

    vessel_coords = np.argwhere(vessel_data > 0)
    vessel_cloud = pv.PolyData(vessel_coords)


    stenosis_cloud = create_stenosis_actor(stenosis_data)


    plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800))


    plotter.subplot(0, 0)
    plotter.add_mesh(vessel_cloud, color='gray', opacity=0.2, point_size=3, render_points_as_spheres=True)
    plotter.add_text("Without Stenosis Points", font_size=12, color='black')


    plotter.subplot(0, 1)
    plotter.add_mesh(vessel_cloud, color='gray', opacity=0.2, point_size=3, render_points_as_spheres=True)
    plotter.add_mesh(stenosis_cloud, scalars='Colors', rgb=True, point_size=15, render_points_as_spheres=True)
    plotter.add_text("With Stenosis Points", font_size=12, color='black')


    plotter.link_views()


    plotter.show()

def visual_main(dir,id):

    patient_id = id
    root_dir = dir
    nii_file = rf'{root_dir}\{patient_id}\label.nii.gz'
    vessel_data = load_vessel_image(nii_file)


    stenosis_file = rf'{root_dir}\{patient_id}\st_area_data.xlsx'
    stenosis_data = pd.read_excel(stenosis_file)

    visualize_vessel_comparison(vessel_data, stenosis_data)

if __name__ == '__main__':
    visual_main()
