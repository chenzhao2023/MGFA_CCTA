# MGFA-CCTA Myocardium Segmentation Pipeline

This repository provides an integrated CTA myocardium preprocessing pipeline. The pipeline slices 3D CTA images into 2D images, performs myocardium segmentation using a pretrained 2D model, merges the predicted 2D masks back into a 3D myocardium mask, and generates the expanded myocardium contour mask.

## Model Weights

The pretrained myocardium segmentation model is not included in this repository because the `.pth` file is larger than GitHub's normal file size limit.

Please download the model weight from Google Drive:

[Download myocardium segmentation model](https://drive.google.com/file/d/1fvckxy5qoMTlImfneBhUd7GYQju4Dj0B/view?usp=sharing)

After downloading, place the `.pth` file in the following default path:

```text
myo_model.pth
```

If you save the weight file elsewhere, specify its path with `--weights` when running the script.

## Input Data Format

The expected CTA data structure is:

```text
..\imgcas
├── case_id_1
│   └── img.nii.gz
├── case_id_2
│   └── img.nii.gz
└── case_id_3
    └── img.nii.gz
```

For example:

```text
..\imgcas\10016975\img.nii.gz
..\imgcas\12019951\img.nii.gz
```

The folder name is automatically used as the `case_id`.

## Environment

Install the required Python packages:

```bash
pip install numpy opencv-python nibabel torch pillow scikit-image tqdm
```

## Usage

If your paths match the default settings, run:

```bash
python cta_myocardium_pipeline.py --device cuda:0
```

The default paths are:

```text
Input CTA directory:
..\imgcas

Model weight:
myo_model.pth

Output directory:
cta_pipeline_output
```

If you need to specify custom paths:

```bash
python cta_myocardium_pipeline.py ^
  --input-dir "..\imgcas" ^
  --model-root "" ^
  --weights "" ^
  --output-dir "" ^
  --device cuda:0
```

To process only one case:

```bash
python cta_myocardium_pipeline.py --case-id 10016975 --device cuda:0
```

## Output

The pipeline generates four output folders:

```text
cta_pipeline_output
├── 01_slices
├── 02_pred_slices
├── 03_myocardium_mask
└── 04_contour
```

The main final results are:

```text
03_myocardium_mask\case_id_label.nii.gz
04_contour\case_id_label.nii.gz
```

## Pipeline Steps

1. Slice the 3D CTA image into 2D PNG images.
2. Predict the myocardium mask for each 2D slice.
3. Merge 2D predictions into a 3D myocardium mask.
4. Generate the expanded myocardium contour using the modified contour dilation logic.

