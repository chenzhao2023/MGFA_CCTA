import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from torch.cuda.amp import autocast, GradScaler
# Avoid duplicate OpenMP runtime crashes on some Windows Python environments.
# Put this before importing numpy/cv2/torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from skimage import measure
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class CaseInfo:
    case_id: str
    image_path: Path
    image_shape: Tuple[int, int, int]
    affine: np.ndarray
    header: nib.Nifti1Header


@dataclass
class SliceRecord:
    case_id: str
    slice_index: int
    path: Path
    original_hw: Tuple[int, int]


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def infer_case_id(path: Path) -> str:
    if path.name in {"img.nii.gz", "img.nii"}:
        return path.parent.name

    stem = strip_nii_suffix(path)
    if stem.endswith("img"):
        stem = stem[:-3]
    if stem.endswith("_"):
        stem = stem[:-1]
    return stem


def find_nifti_images(input_dir: Path, case_ids: Optional[Iterable[str]] = None) -> List[Path]:
    wanted = set(str(i) for i in case_ids) if case_ids else None

    nested_paths = []
    for case_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        gz_path = case_dir / "img.nii.gz"
        nii_path = case_dir / "img.nii"
        if gz_path.is_file():
            nested_paths.append(gz_path)
        elif nii_path.is_file():
            nested_paths.append(nii_path)

    flat_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))
    )

    paths = nested_paths if nested_paths else flat_paths
    if wanted is None:
        return paths
    return [p for p in paths if infer_case_id(p) in wanted]


def normalize_to_uint8(image_2d: np.ndarray) -> np.ndarray:
    image_2d = np.nan_to_num(image_2d).astype(np.float32)
    min_value = float(image_2d.min())
    max_value = float(image_2d.max())
    if max_value <= min_value:
        return np.zeros(image_2d.shape, dtype=np.uint8)
    image_2d = (image_2d - min_value) / (max_value - min_value)
    return np.clip(image_2d * 255.0, 0, 255).astype(np.uint8)


def split_cta_to_slices(input_dir: Path, slice_dir: Path, case_ids: Optional[List[str]] = None) -> Tuple[List[CaseInfo], List[SliceRecord]]:
    slice_dir.mkdir(parents=True, exist_ok=True)
    cases: List[CaseInfo] = []
    records: List[SliceRecord] = []

    image_paths = find_nifti_images(input_dir, case_ids)
    if not image_paths:
        raise FileNotFoundError(f"No NIfTI images found in {input_dir}")

    for image_path in tqdm(image_paths, desc="Step 1/4 slicing CTA"):
        nii = nib.load(str(image_path))
        image = nii.get_fdata()
        if image.ndim != 3:
            raise ValueError(f"{image_path} is not a 3D image, got shape {image.shape}")

        case_id = infer_case_id(image_path)
        cases.append(CaseInfo(case_id, image_path, image.shape, nii.affine, nii.header.copy()))

        for slice_index in range(image.shape[2]):
            slice_2d = image[:, :, slice_index]
            rotated = cv2.rotate(normalize_to_uint8(slice_2d), cv2.ROTATE_90_COUNTERCLOCKWISE)
            out_path = slice_dir / f"{case_id}_image_slice_{slice_index}.png"
            cv2.imwrite(str(out_path), rotated)
            records.append(SliceRecord(case_id, slice_index, out_path, image.shape[:2]))

    return cases, records


class CTASliceDataset(Dataset):
    def __init__(self, records: List[SliceRecord], input_size: int):
        self.records = records
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        # image = Image.open(record.path).convert("L")
        # image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        image = cv2.imread(record.path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor, index


def import_model(model_name: str, model_root: Optional[Path]):
    if model_root is not None:
        sys.path.insert(0, str(model_root.resolve()))

    if model_name == "unet":
        from models.unet import U_Net
        return U_Net()
    if model_name == "ultralight":
        from models.UltraLight_VM_UNet import UltraLight_VM_UNet
        return UltraLight_VM_UNet(1, 1)
    raise ValueError(f"Unsupported model name: {model_name}")


def load_segmentation_model(model_name: str, model_root: Optional[Path], weights: Path, device: torch.device):
    model = import_model(model_name, model_root).to(device)
    checkpoint = torch.load(str(weights), map_location=device)
    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    checkpoint = {
        key.replace("module.", "", 1): value
        for key, value in checkpoint.items()
    }
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict_myocardium_slices(
    records: List[SliceRecord],
    pred_dir: Path,
    model_name: str,
    model_root: Optional[Path],
    weights: Path,
    device_name: str,
    input_size: int,
    batch_size: int,
    num_workers: int,
    threshold: float,
) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if device_name != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = load_segmentation_model(model_name, model_root, weights, device)
    loader = DataLoader(
        CTASliceDataset(records, input_size=input_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    with torch.no_grad():
        for images, indices in tqdm(loader, desc="Step 2/4 segmenting myocardium"):
            images = images.to(device)

            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            if outputs.min().item() < 0 or outputs.max().item() > 1:
                print(0)
                outputs = torch.sigmoid(outputs)
            preds = (outputs > threshold).cpu().numpy().astype(int)

            for batch_pos, record_index in enumerate(indices.tolist()):
                record = records[record_index]
                pred = preds[batch_pos, 0]
                pred[np.where(pred == 1)] = 255
                out_name = f"{record.path.name}_pred.png"
                cv2.imwrite(str(pred_dir / out_name), pred)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = measure.label(mask > 0, return_num=True)
    if num == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    regions = measure.regionprops(labeled)
    largest_label = max(regions, key=lambda item: item.area).label
    return (labeled == largest_label).astype(np.uint8)


def merge_slices_to_3d(cases: List[CaseInfo], records: List[SliceRecord], pred_dir: Path, mask_dir: Path, largest_only: bool) -> Dict[str, Path]:
    mask_dir.mkdir(parents=True, exist_ok=True)
    by_case: Dict[str, List[SliceRecord]] = {}
    for record in records:
        by_case.setdefault(record.case_id, []).append(record)

    case_by_id = {case.case_id: case for case in cases}
    output_paths: Dict[str, Path] = {}

    for case_id, case_records in tqdm(by_case.items(), desc="Step 3/4 merging 3D masks"):
        case = case_by_id[case_id]
        volume = np.zeros(case.image_shape, dtype=np.uint8)

        for record in sorted(case_records, key=lambda item: item.slice_index):
            pred_path = pred_dir / f"{record.path.name}_pred.png"
            pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            if pred is None:
                raise FileNotFoundError(f"Missing prediction: {pred_path}")

            original_h, original_w = record.original_hw
            pred = cv2.resize(pred, (original_h, original_w), interpolation=cv2.INTER_NEAREST)
            pred = cv2.rotate(pred, cv2.ROTATE_90_CLOCKWISE)
            volume[:, :, record.slice_index] = (pred > 0).astype(np.uint8)

        if largest_only:
            volume = keep_largest_component(volume)

        out_path = mask_dir / f"{case_id}_label.nii.gz"
        nib.save(nib.Nifti1Image(volume, case.affine, case.header), str(out_path))
        output_paths[case_id] = out_path

    return output_paths


def build_myocardium_contour(slice_2d: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    binary = (slice_2d > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return (binary > 0).astype(np.uint8)

    contour_image = np.zeros_like(binary)
    contour_image = np.ascontiguousarray(contour_image, dtype=np.uint8)

    for contour in contours:
        leftmost_idx = contour[:, :, 0].argmin()
        bottommost_idx = contour[:, :, 1].argmax()

        new_contour = []
        start_idx = min(leftmost_idx, bottommost_idx)
        end_idx = max(leftmost_idx, bottommost_idx)
        for point_idx in range(len(contour)):
            if not (start_idx <= point_idx <= end_idx):
                new_contour.append(contour[point_idx])

        if len(new_contour) == 0:
            continue

        new_contour = np.array(new_contour, dtype=np.int32)
        cv2.drawContours(contour_image, [new_contour], -1, 255, thickness=1)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_contour = cv2.dilate(contour_image, kernel, iterations=iterations)
    return (dilated_contour > 0).astype(np.uint8)


def get_contour_slice(
    slice_2d: np.ndarray,
    contour_kernel_size: int,
    iterations: int,
    prefer_tianchong: bool,
) -> np.ndarray:
    slice_2d = (slice_2d > 0).astype(np.uint8) * 255
    if prefer_tianchong:
        try:
            from tianchong import get_dilated_contour
            contour = get_dilated_contour(slice_2d)
            return (contour > 0).astype(np.uint8)
        except ImportError:
            pass

    return build_myocardium_contour(slice_2d, contour_kernel_size, iterations)


def expand_contours(
    mask_paths: Dict[str, Path],
    contour_dir: Path,
    contour_kernel_size: int,
    iterations: int,
    prefer_tianchong: bool,
) -> None:
    contour_dir.mkdir(parents=True, exist_ok=True)

    for case_id, mask_path in tqdm(mask_paths.items(), desc="Step 4/4 expanding contours"):
        nii = nib.load(str(mask_path))
        mask = nii.get_fdata()
        contour_volume = np.zeros_like(mask, dtype=np.uint8)

        for slice_index in range(mask.shape[2]):
            slice_2d = mask[:, :, slice_index]
            if np.sum(slice_2d) == 0:
                continue
            flipped = cv2.flip(slice_2d.astype(np.uint8), 0)
            contour = get_contour_slice(flipped, contour_kernel_size, iterations, prefer_tianchong)
            contour_volume[:, :, slice_index] = cv2.flip(contour, 0)

        out_path = contour_dir / f"{case_id}_label.nii.gz"
        nib.save(nib.Nifti1Image(contour_volume, nii.affine, nii.header), str(out_path))


def read_case_ids(case_id_file: Optional[Path], case_ids: Optional[List[str]]) -> Optional[List[str]]:
    ids: List[str] = []
    if case_ids:
        ids.extend(case_ids)
    if case_id_file:
        ids.extend(line.strip() for line in case_id_file.read_text(encoding="utf-8").splitlines() if line.strip())
    return ids or None


def parse_args():
    parser = argparse.ArgumentParser(
        description="CTA slicing -> myocardium segmentation -> 3D mask merging -> contour expansion pipeline."
    )
    parser.add_argument("--input-dir", default=Path(r"D:\imgcas"), type=Path, help="CTA directory, e.g. D:/imgcas/case_id/img.nii.gz")
    parser.add_argument("--weights", default=Path(r"myo_model.pth"), type=Path, help="2D myocardium segmentation model weights (.pth)")
    parser.add_argument("--output-dir", default=Path(r"cta_pipeline_output"), type=Path, help="Pipeline output directory")
    parser.add_argument("--model-name", default="unet", help="Model architecture")
    parser.add_argument("--model-root", default=Path(r""), type=Path, help="Project root that contains the models/ directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, cuda:1, ...")
    parser.add_argument("--input-size", default=512, type=int, help="2D model input size")
    parser.add_argument("--batch-size", default=36, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--case-id", action="append", dest="case_ids", help="Run one case id. Can be used multiple times")
    parser.add_argument("--case-id-file", type=Path, help="Text file containing one case id per line")
    parser.add_argument("--no-largest-component", action="store_true",default=False, help="Do not keep only the largest 3D connected component")
    parser.add_argument("--prefer-tianchong", action="store_true", help="Use tianchong.get_dilated_contour if available")
    parser.add_argument("--contour-kernel-size", default=50, type=int, help="Kernel size for dilating the modified contour")
    parser.add_argument("--iterations", default=1, type=int, help="Morphology iterations")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir: Path = args.output_dir
    slice_dir = output_dir / "01_slices"
    pred_dir = output_dir / "02_pred_slices"
    mask_dir = output_dir / "03_myocardium_mask"
    contour_dir = output_dir / "04_contour"

    case_ids = read_case_ids(args.case_id_file, args.case_ids)
    cases, records = split_cta_to_slices(args.input_dir, slice_dir, case_ids)
    predict_myocardium_slices(
        records=records,
        pred_dir=pred_dir,
        model_name=args.model_name,
        model_root=args.model_root,
        weights=args.weights,
        device_name=args.device,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )
    mask_paths = merge_slices_to_3d(
        cases=cases,
        records=records,
        pred_dir=pred_dir,
        mask_dir=mask_dir,
        largest_only=not args.no_largest_component,
    )
    expand_contours(
        mask_paths=mask_paths,
        contour_dir=contour_dir,
        contour_kernel_size=args.contour_kernel_size,
        iterations=args.iterations,
        prefer_tianchong=args.prefer_tianchong,
    )
    print(f"Done. Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
