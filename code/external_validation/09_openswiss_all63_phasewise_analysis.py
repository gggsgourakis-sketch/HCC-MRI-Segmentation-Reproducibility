#!/usr/bin/env python3
"""
Phase-wise external HCC segmentation analysis using the completed ATLAS nnU-Net ensemble.

Cohorts
-------
1. LiverHccSeg: arterial, portal-venous, delayed and native/precontrast MRI.
   The primary analysis is restricted to tumour-positive cases.
2. OpenSwissHCC: arterial, venous, delayed and native water-reconstruction T1w MRI.
   HCC-only phase-specific references are rebuilt from participants.tsv and derivatives.zip.
   The primary analysis excludes cases with any missing phase-specific HCC reference mask.

The script:
- prepares nnU-Net input folders using symbolic links;
- builds phase-specific reference masks;
- runs the existing Dataset801 five-fold ensemble without fine-tuning;
- calculates case Dice, IoU, volume error, HD95 and ASSD;
- calculates lesion precision, recall, F1, matched-lesion Dice and FP/case;
- exports CSV, JSON, text summaries and 600-dpi PNG figures.

No source files are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta

CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)
PHASES = ("arterial", "portal_venous", "delayed", "native")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run phase-wise external MRI evaluation using the completed ATLAS nnU-Net ensemble."
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("./hcc_openswiss_all63_work"),
        help="Linux filesystem directory for inputs, predictions and reference masks.",
    )
    p.add_argument(
        "--export-dir",
        type=Path,
        default=Path("./HCC_OpenSwiss_All63_Results"),
        help="Small final result package written to Windows Desktop.",
    )
    p.add_argument(
        "--liver-root",
        type=Path,
        default=Path("/mnt/d/DESKTOP/HCC public databases/ZENODO-LiverHccSeg/7957516"),
    )
    p.add_argument(
        "--openswiss-root",
        type=Path,
        default=Path("./source_unpacked"),
    )
    p.add_argument(
        "--openswiss-participants",
        type=Path,
        default=Path(
            "/mnt/d/DESKTOP/HCC public databases/ZENODO-OpenSwissHCC/17517079/participants.tsv"
        ),
    )
    p.add_argument(
        "--openswiss-derivatives-zip",
        type=Path,
        default=Path(
            "/mnt/d/DESKTOP/HCC public databases/ZENODO-OpenSwissHCC/17517079/derivatives.zip"
        ),
    )
    p.add_argument(
        "--openswiss-subset-predictions",
        type=Path,
        default=Path("./preds_raw_venous"),
        help="Legacy option retained for command compatibility; ignored in the all-63 HCC-positive analysis.",
    )
    p.add_argument(
        "--predict-exe",
        type=Path,
        default=Path("nnUNetv2_predict"),
    )
    p.add_argument(
        "--nnunet-raw",
        type=Path,
        default=Path("./nnunet"),
    )
    p.add_argument(
        "--nnunet-preprocessed",
        type=Path,
        default=Path("./nnunet/preprocessed"),
    )
    p.add_argument(
        "--nnunet-results",
        type=Path,
        default=Path("./nnunet/results"),
    )
    p.add_argument("--dataset", default="Dataset801_HCCMRI_ATLAS")
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--trainer", default="nnUNetTrainer")
    p.add_argument("--plans", default="nnUNetPlans")
    p.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    p.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    p.add_argument(
        "--cohorts", nargs="+", choices=("liver", "openswiss"), default=["liver", "openswiss"]
    )
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260725)
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare inputs and references but do not run nnU-Net or calculate final metrics.",
    )
    p.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Do not prepare or predict; evaluate existing prediction folders.",
    )
    p.add_argument(
        "--overwrite-predictions",
        action="store_true",
        help="Pass overwrite mode to nnU-Net and rerun phases with existing predictions.",
    )
    p.add_argument(
        "--disable-tta",
        action="store_true",
        help="Disable nnU-Net mirroring. Default preserves standard nnU-Net inference behaviour.",
    )
    return p.parse_args()


def require(path: Path, kind: str) -> None:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def safe_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    else:
        fields = fieldnames
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_binary(path: Path) -> Tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    return image, np.asarray(image.dataobj) > 0


def align_image(moving: nib.Nifti1Image, reference: nib.Nifti1Image) -> nib.Nifti1Image:
    if moving.shape == reference.shape and np.allclose(moving.affine, reference.affine, atol=1e-4):
        return moving
    return resample_from_to(moving, reference, order=0)


def save_union_mask(mask_paths: Sequence[Path], reference_image: Path, destination: Path) -> int:
    ref = nib.load(str(reference_image))
    union = np.zeros(ref.shape, dtype=bool)
    for mask_path in mask_paths:
        image = nib.load(str(mask_path))
        image = align_image(image, ref)
        union |= np.asarray(image.dataobj) > 0
    out = nib.Nifti1Image(union.astype(np.uint8), ref.affine, ref.header.copy())
    out.set_data_dtype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(destination))
    return int(union.sum())


def save_union_images(
    mask_images: Sequence[nib.Nifti1Image],
    reference_image: Path,
    destination: Path,
) -> int:
    ref = nib.load(str(reference_image))
    union = np.zeros(ref.shape, dtype=bool)
    for image in mask_images:
        image = align_image(image, ref)
        union |= np.asarray(image.dataobj) > 0
    out = nib.Nifti1Image(union.astype(np.uint8), ref.affine, ref.header.copy())
    out.set_data_dtype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(destination))
    return int(union.sum())


def candidate_file(directory: Path, names: Sequence[str]) -> Optional[Path]:
    lower_map = {p.name.lower(): p for p in directory.glob("*.nii.gz")}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


LIVER_PHASE_NAMES = {
    "arterial": ("art.nii.gz", "arterial.nii.gz", "arterial_phase.nii.gz"),
    "portal_venous": (
        "pv.nii.gz",
        "portal.nii.gz",
        "venous.nii.gz",
        "portal_venous.nii.gz",
    ),
    "delayed": ("delay.nii.gz", "delayed.nii.gz", "del.nii.gz"),
    "native": (
        "pre.nii.gz",
        "native.nii.gz",
        "precontrast.nii.gz",
        "pre_contrast.nii.gz",
    ),
}


def discover_liver_case_dirs(root: Path) -> List[Path]:
    phase_names = {name.lower() for names in LIVER_PHASE_NAMES.values() for name in names}
    case_dirs = set()
    for path in root.rglob("*.nii.gz"):
        if path.name.lower() in phase_names:
            case_dirs.add(path.parent)
    return sorted(case_dirs, key=lambda p: str(p))


def prepare_liver(args: argparse.Namespace) -> List[dict]:
    require(args.liver_root, "dir")
    case_dirs = discover_liver_case_dirs(args.liver_root)
    if not case_dirs:
        raise RuntimeError(f"No LiverHccSeg phase files found under {args.liver_root}")

    rows: List[dict] = []
    for idx, case_dir in enumerate(case_dirs, start=1):
        case_id = f"LiverHccSeg_{idx:03d}"
        subject_id = case_dir.parent.name
        study_date = case_dir.name
        tumour_masks = sorted(case_dir.glob("rater1_tumor*.nii.gz"))

        for phase in args.phases:
            image = candidate_file(case_dir, LIVER_PHASE_NAMES[phase])
            if image is None:
                rows.append(
                    {
                        "cohort": "LiverHccSeg",
                        "phase": phase,
                        "case_id": case_id,
                        "subject_id": subject_id,
                        "study_date": study_date,
                        "source_image": "",
                        "reference_mask": "",
                        "reference_complete": 0,
                        "documented_lesions": len(tumour_masks),
                        "available_phase_masks": len(tumour_masks),
                        "status": "phase image missing",
                    }
                )
                continue

            image_dir = args.work_dir / "LiverHccSeg" / phase / "imagesTs"
            label_dir = args.work_dir / "LiverHccSeg" / phase / "labelsTs"
            input_path = image_dir / f"{case_id}_0000.nii.gz"
            label_path = label_dir / f"{case_id}.nii.gz"
            safe_symlink(image, input_path)
            gt_voxels = save_union_mask(tumour_masks, image, label_path)

            rows.append(
                {
                    "cohort": "LiverHccSeg",
                    "phase": phase,
                    "case_id": case_id,
                    "subject_id": subject_id,
                    "study_date": study_date,
                    "source_image": str(image),
                    "reference_mask": str(label_path),
                    "reference_complete": 1,
                    "documented_lesions": len(tumour_masks),
                    "available_phase_masks": len(tumour_masks),
                    "reference_voxels": gt_voxels,
                    "status": "prepared",
                }
            )
    return rows


def read_participants(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def lesion_index(value: object) -> Optional[int]:
    text = clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def openswiss_subjects_from_predictions(path: Path) -> List[str]:
    require(path, "dir")
    subjects = []
    for prediction in sorted(path.glob("OpenSwissHCC_*.nii.gz")):
        match = re.fullmatch(r"OpenSwissHCC_(\d{3})\.nii\.gz", prediction.name)
        if match:
            subjects.append(f"sub-{match.group(1)}")
    subjects = sorted(set(subjects))
    if not subjects:
        raise RuntimeError(f"No OpenSwissHCC prediction files found in {path}")
    return subjects


def openswiss_phase_candidates(subject_dir: Path, phase: str) -> List[Path]:
    dyn = subject_dir / "dyn"
    if phase == "portal_venous":
        patterns = [f"{subject_dir.name}_acq-water_phase-venous_T1w.nii.gz"]
    elif phase == "native":
        patterns = [f"{subject_dir.name}_acq-water_phase-native_T1w.nii.gz"]
    elif phase == "delayed":
        patterns = [f"{subject_dir.name}_acq-water_phase-delayed_T1w.nii.gz"]
    else:
        patterns = [
            f"{subject_dir.name}_acq-water_phase-arterial_T1w.nii.gz",
            f"{subject_dir.name}_acq-water_phase-arterial-TTC-2_T1w.nii.gz",
            f"{subject_dir.name}_acq-water_phase-arterial-TTC-1_T1w.nii.gz",
            f"{subject_dir.name}_acq-water_phase-arterial-TTC-3_T1w.nii.gz",
        ]
    return [dyn / name for name in patterns if (dyn / name).is_file()]


def mask_member_for_image(
    names: Sequence[str], image_path: Path, lesion_idx: int
) -> Optional[str]:
    image_stem = image_path.name.removesuffix(".nii.gz")
    filename = f"{image_stem}-L{lesion_idx}_seg.nii.gz"
    candidates = [
        name for name in names
        if name.endswith("/" + filename) or name == filename
    ]
    if not candidates:
        lower = filename.lower()
        candidates = [name for name in names if name.lower().endswith(lower)]
    if not candidates:
        return None
    candidates.sort(
        key=lambda name: (
            "manual_lesion_annotations" not in name,
            len(name),
            name,
        )
    )
    return candidates[0]


def choose_openswiss_image(
    candidates: Sequence[Path],
    lesion_indices: Sequence[int],
    archive_names: Sequence[str],
) -> Tuple[Optional[Path], List[Optional[str]]]:
    if not candidates:
        return None, []

    scored = []
    for rank, image in enumerate(candidates):
        members = [
            mask_member_for_image(archive_names, image, idx)
            for idx in lesion_indices
        ]
        available = sum(member is not None for member in members)
        # Choose the candidate with the largest complete annotation count.
        # The input ordering gives the prespecified tie-break:
        # plain arterial, TTC-2, TTC-1, TTC-3.
        scored.append((available, -rank, image, members))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    _, _, image, members = scored[0]
    return image, members


def read_nifti_from_zip(
    archive: zipfile.ZipFile, member: str, temp_root: Path, unique_name: str
) -> nib.Nifti1Image:
    destination = temp_root / unique_name
    with archive.open(member) as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return nib.load(str(destination))


def prepare_openswiss(args: argparse.Namespace) -> List[dict]:
    require(args.openswiss_root, "dir")
    require(args.openswiss_participants, "file")
    require(args.openswiss_derivatives_zip, "file")

    participants = read_participants(args.openswiss_participants)
    hcc_indices: Dict[str, List[int]] = defaultdict(list)
    for row in participants:
        if clean(row.get("HCC")) != "1":
            continue
        subject = clean(row.get("ID"))
        idx = lesion_index(row.get("Lesion index"))
        if subject and idx is not None:
            hcc_indices[subject].append(idx)

    # Editor-directed correction: define eligibility from clinical HCC metadata,
    # not from the existence of an old venous prediction file.
    subset = sorted(hcc_indices.keys())
    if len(subset) != 63:
        print(f"[WARN] Expected 63 HCC-positive subjects from participants.tsv; found {len(subset)}")
    else:
        print("[OK] OpenSwissHCC eligibility: all 63 HCC-positive subjects from participants.tsv")

    rows: List[dict] = []
    with zipfile.ZipFile(args.openswiss_derivatives_zip, "r") as archive:
        names = archive.namelist()
        with tempfile.TemporaryDirectory(prefix="openswiss_phase_refs_") as temp_dir:
            temp_root = Path(temp_dir)

            for subject in subset:
                subject_dir = args.openswiss_root / subject
                indices = sorted(set(hcc_indices.get(subject, [])))
                case_id = f"OpenSwissHCC_{subject.split('-')[1]}"

                for phase in args.phases:
                    candidates = openswiss_phase_candidates(subject_dir, phase)
                    image, members = choose_openswiss_image(candidates, indices, names)
                    if image is None:
                        rows.append(
                            {
                                "cohort": "OpenSwissHCC",
                                "phase": phase,
                                "case_id": case_id,
                                "subject_id": subject,
                                "source_image": "",
                                "reference_mask": "",
                                "reference_complete": 0,
                                "documented_lesions": len(indices),
                                "available_phase_masks": 0,
                                "status": "phase image missing",
                            }
                        )
                        continue

                    valid_members = [member for member in members if member is not None]
                    missing = len(indices) - len(valid_members)
                    mask_images = []
                    for lesion_idx, member in zip(indices, members):
                        if member is None:
                            continue
                        mask_images.append(
                            read_nifti_from_zip(
                                archive,
                                member,
                                temp_root,
                                f"{subject}_{phase}_L{lesion_idx}.nii.gz",
                            )
                        )

                    image_dir = args.work_dir / "OpenSwissHCC" / phase / "imagesTs"
                    label_dir = args.work_dir / "OpenSwissHCC" / phase / "labelsTs"
                    input_path = image_dir / f"{case_id}_0000.nii.gz"
                    label_path = label_dir / f"{case_id}.nii.gz"
                    safe_symlink(image, input_path)
                    gt_voxels = save_union_images(mask_images, image, label_path)

                    rows.append(
                        {
                            "cohort": "OpenSwissHCC",
                            "phase": phase,
                            "case_id": case_id,
                            "subject_id": subject,
                            "source_image": str(image),
                            "arterial_variant": (
                                image.name.removesuffix(".nii.gz")
                                if phase == "arterial"
                                else ""
                            ),
                            "reference_mask": str(label_path),
                            "reference_complete": int(missing == 0),
                            "documented_lesions": len(indices),
                            "available_phase_masks": len(valid_members),
                            "missing_phase_masks": missing,
                            "reference_voxels": gt_voxels,
                            "status": "prepared" if missing == 0 else "prepared_incomplete_reference",
                        }
                    )
    return rows


def run_prediction(
    args: argparse.Namespace,
    cohort: str,
    phase: str,
) -> None:
    input_dir = args.work_dir / cohort / phase / "imagesTs"
    prediction_dir = args.work_dir / cohort / phase / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*_0000.nii.gz"))
    if not input_files:
        print(f"[SKIP] No prepared inputs for {cohort} / {phase}")
        return

    existing = sorted(prediction_dir.glob("*.nii.gz"))
    if len(existing) >= len(input_files) and not args.overwrite_predictions:
        print(
            f"[SKIP] {cohort} / {phase}: {len(existing)} predictions already exist "
            f"for {len(input_files)} inputs."
        )
        return

    cmd = [
        str(args.predict_exe),
        "-i", str(input_dir),
        "-o", str(prediction_dir),
        "-d", args.dataset,
        "-c", args.configuration,
        "-f", *args.folds,
        "-tr", args.trainer,
        "-p", args.plans,
    ]
    if args.overwrite_predictions:
        cmd.append("--continue_prediction")
    if args.disable_tta:
        cmd.append("--disable_tta")

    env = os.environ.copy()
    env["nnUNet_raw"] = str(args.nnunet_raw)
    env["nnUNet_preprocessed"] = str(args.nnunet_preprocessed)
    env["nnUNet_results"] = str(args.nnunet_results)

    print(f"\n[RUN] {cohort} / {phase}")
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    n_gt = int(gt.sum())
    n_pred = int(pred.sum())
    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt + n_pred == 0:
        return 0.0
    return 2.0 * int(np.logical_and(gt, pred).sum()) / (n_gt + n_pred)


def iou_score(gt: np.ndarray, pred: np.ndarray) -> float:
    union = int(np.logical_or(gt, pred).sum())
    return 1.0 if union == 0 else int(np.logical_and(gt, pred).sum()) / union


def surface_distances(
    source: np.ndarray, target: np.ndarray, spacing: Sequence[float]
) -> np.ndarray:
    source_surface = np.logical_xor(
        source,
        ndimage.binary_erosion(source, structure=CONNECTIVITY, border_value=0),
    )
    target_surface = np.logical_xor(
        target,
        ndimage.binary_erosion(target, structure=CONNECTIVITY, border_value=0),
    )
    if not source_surface.any() or not target_surface.any():
        return np.asarray([], dtype=float)
    distance_map = ndimage.distance_transform_edt(
        ~target_surface, sampling=tuple(float(v) for v in spacing)
    )
    return distance_map[source_surface]


def boundary_metrics(
    gt: np.ndarray, pred: np.ndarray, spacing: Sequence[float]
) -> Tuple[Optional[float], Optional[float]]:
    if not gt.any() or not pred.any():
        return None, None
    a = surface_distances(gt, pred, spacing)
    b = surface_distances(pred, gt, spacing)
    if a.size == 0 or b.size == 0:
        return None, None
    all_distances = np.concatenate([a, b])
    return float(np.percentile(all_distances, 95)), float((a.mean() + b.mean()) / 2.0)


def component_masks(mask: np.ndarray) -> List[np.ndarray]:
    labels, count = ndimage.label(mask, structure=CONNECTIVITY)
    return [labels == idx for idx in range(1, count + 1)]


def match_components(
    gt_components: List[np.ndarray], pred_components: List[np.ndarray]
) -> Tuple[List[dict], int, int, int]:
    n_gt = len(gt_components)
    n_pred = len(pred_components)
    if n_gt == 0 or n_pred == 0:
        rows = [
            {
                "gt_component": idx,
                "gt_voxels": int(mask.sum()),
                "detected": 0,
                "matched_pred_component": "",
                "matched_pred_voxels": "",
                "lesion_dice": 0.0,
                "lesion_iou": 0.0,
            }
            for idx, mask in enumerate(gt_components, start=1)
        ]
        return rows, 0, n_pred, n_gt

    iou_matrix = np.zeros((n_gt, n_pred), dtype=float)
    dice_matrix = np.zeros((n_gt, n_pred), dtype=float)
    for i, gt in enumerate(gt_components):
        gt_n = int(gt.sum())
        for j, pred in enumerate(pred_components):
            intersection = int(np.logical_and(gt, pred).sum())
            if intersection == 0:
                continue
            pred_n = int(pred.sum())
            iou_matrix[i, j] = intersection / (gt_n + pred_n - intersection)
            dice_matrix[i, j] = 2.0 * intersection / (gt_n + pred_n)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matches = {
        int(i): int(j)
        for i, j in zip(row_ind, col_ind)
        if iou_matrix[i, j] > 0
    }

    rows = []
    for i, gt in enumerate(gt_components):
        if i in matches:
            j = matches[i]
            rows.append(
                {
                    "gt_component": i + 1,
                    "gt_voxels": int(gt.sum()),
                    "detected": 1,
                    "matched_pred_component": j + 1,
                    "matched_pred_voxels": int(pred_components[j].sum()),
                    "lesion_dice": float(dice_matrix[i, j]),
                    "lesion_iou": float(iou_matrix[i, j]),
                }
            )
        else:
            rows.append(
                {
                    "gt_component": i + 1,
                    "gt_voxels": int(gt.sum()),
                    "detected": 0,
                    "matched_pred_component": "",
                    "matched_pred_voxels": "",
                    "lesion_dice": 0.0,
                    "lesion_iou": 0.0,
                }
            )

    tp = len(matches)
    return rows, tp, n_pred - tp, n_gt - tp


def bootstrap_ci(
    values: Sequence[float], replicates: int, seed: int
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def exact_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def finite_values(values: Iterable[object]) -> List[float]:
    result = []
    for value in values:
        if value in ("", None):
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def evaluate(args: argparse.Namespace, mapping_rows: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    case_rows: List[dict] = []
    lesion_rows: List[dict] = []

    for mapping in mapping_rows:
        if not mapping.get("source_image") or not mapping.get("reference_mask"):
            continue
        cohort = mapping["cohort"]
        phase = mapping["phase"]
        case_id = mapping["case_id"]
        prediction = args.work_dir / cohort / phase / "predictions" / f"{case_id}.nii.gz"
        reference = Path(mapping["reference_mask"])
        if not prediction.is_file():
            continue

        ref_img, gt = load_binary(reference)
        pred_img, pred = load_binary(prediction)
        pred_img = align_image(pred_img, ref_img)
        pred = np.asarray(pred_img.dataobj) > 0
        spacing = tuple(float(v) for v in ref_img.header.get_zooms()[:3])
        voxel_ml = float(np.prod(spacing)) / 1000.0

        gt_voxels = int(gt.sum())
        pred_voxels = int(pred.sum())
        gt_ml = gt_voxels * voxel_ml
        pred_ml = pred_voxels * voxel_ml
        signed_error = pred_ml - gt_ml
        relative_error = 100.0 * signed_error / gt_ml if gt_ml > 0 else None
        hd95, assd = boundary_metrics(gt, pred, spacing)

        gt_components = component_masks(gt)
        pred_components = component_masks(pred)
        matches, tp, fp, fn = match_components(gt_components, pred_components)

        for row in matches:
            lesion_rows.append(
                {
                    "cohort": cohort,
                    "phase": phase,
                    "case_id": case_id,
                    "subject_id": mapping.get("subject_id", ""),
                    **row,
                }
            )

        dice = dice_score(gt, pred)
        if gt_voxels == 0:
            category = "empty reference"
        elif tp == 0:
            category = "complete detection failure"
        elif dice < 0.50:
            category = "detected but poor contour agreement"
        else:
            category = "moderate-to-good overlap"

        case_rows.append(
            {
                "cohort": cohort,
                "phase": phase,
                "case_id": case_id,
                "subject_id": mapping.get("subject_id", ""),
                "reference_complete": int(mapping.get("reference_complete", 0)),
                "documented_lesions": int(mapping.get("documented_lesions", 0)),
                "available_phase_masks": int(mapping.get("available_phase_masks", 0)),
                "gt_components": len(gt_components),
                "pred_components": len(pred_components),
                "lesion_tp": tp,
                "lesion_fp": fp,
                "lesion_fn": fn,
                "gt_voxels": gt_voxels,
                "pred_voxels": pred_voxels,
                "gt_volume_ml": gt_ml,
                "pred_volume_ml": pred_ml,
                "signed_volume_error_ml": signed_error,
                "absolute_volume_error_ml": abs(signed_error),
                "relative_volume_error_pct": "" if relative_error is None else relative_error,
                "dice": dice,
                "iou": iou_score(gt, pred),
                "hd95_mm": "" if hd95 is None else hd95,
                "assd_mm": "" if assd is None else assd,
                "error_category": category,
            }
        )

    summaries: List[dict] = []
    for cohort in sorted({row["cohort"] for row in case_rows}):
        for phase in PHASES:
            all_phase = [
                row for row in case_rows
                if row["cohort"] == cohort and row["phase"] == phase
            ]
            if not all_phase:
                continue

            if cohort == "LiverHccSeg":
                primary = [row for row in all_phase if row["gt_voxels"] > 0]
                primary_definition = "tumour-positive cases"
            else:
                primary = [
                    row for row in all_phase
                    if row["reference_complete"] == 1 and row["gt_voxels"] > 0
                ]
                primary_definition = "complete phase-specific HCC reference"

            if not primary:
                continue

            primary_ids = {row["case_id"] for row in primary}
            primary_lesions = [
                row for row in lesion_rows
                if row["cohort"] == cohort
                and row["phase"] == phase
                and row["case_id"] in primary_ids
            ]
            dice_values = [float(row["dice"]) for row in primary]
            ci_low, ci_high = bootstrap_ci(
                dice_values, args.bootstrap_replicates, args.seed
            )
            tp = sum(int(row["lesion_tp"]) for row in primary)
            fp = sum(int(row["lesion_fp"]) for row in primary)
            fn = sum(int(row["lesion_fn"]) for row in primary)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) else 0.0
            )
            precision_ci = exact_ci(tp, tp + fp) if (tp + fp) else (0.0, 1.0)
            recall_ci = exact_ci(tp, tp + fn) if (tp + fn) else (0.0, 1.0)
            matched_dice = [
                float(row["lesion_dice"])
                for row in primary_lesions
                if int(row["detected"]) == 1
            ]
            hd_values = finite_values(row["hd95_mm"] for row in primary)
            assd_values = finite_values(row["assd_mm"] for row in primary)
            abs_volume = finite_values(row["absolute_volume_error_ml"] for row in primary)
            rel_volume = finite_values(row["relative_volume_error_pct"] for row in primary)

            summaries.append(
                {
                    "cohort": cohort,
                    "phase": phase,
                    "primary_definition": primary_definition,
                    "prepared_cases": len(all_phase),
                    "primary_cases": len(primary),
                    "mean_dice": float(np.mean(dice_values)),
                    "bootstrap_95_ci_low": ci_low,
                    "bootstrap_95_ci_high": ci_high,
                    "median_dice": float(np.median(dice_values)),
                    "q1_dice": float(np.quantile(dice_values, 0.25)),
                    "q3_dice": float(np.quantile(dice_values, 0.75)),
                    "dice_gt_0": sum(value > 0 for value in dice_values),
                    "dice_ge_0_50": sum(value >= 0.50 for value in dice_values),
                    "gt_components": sum(int(row["gt_components"]) for row in primary),
                    "pred_components": sum(int(row["pred_components"]) for row in primary),
                    "lesion_tp": tp,
                    "lesion_fp": fp,
                    "lesion_fn": fn,
                    "lesion_precision": precision,
                    "lesion_precision_ci_low": precision_ci[0],
                    "lesion_precision_ci_high": precision_ci[1],
                    "lesion_recall": recall,
                    "lesion_recall_ci_low": recall_ci[0],
                    "lesion_recall_ci_high": recall_ci[1],
                    "lesion_f1": f1,
                    "false_positives_per_case": fp / len(primary),
                    "median_matched_lesion_dice": (
                        float(np.median(matched_dice)) if matched_dice else ""
                    ),
                    "median_absolute_volume_error_ml": (
                        float(np.median(abs_volume)) if abs_volume else ""
                    ),
                    "median_absolute_relative_volume_error_pct": (
                        float(np.median(np.abs(rel_volume))) if rel_volume else ""
                    ),
                    "boundary_metric_cases": len(hd_values),
                    "median_hd95_mm": float(np.median(hd_values)) if hd_values else "",
                    "median_assd_mm": float(np.median(assd_values)) if assd_values else "",
                    "complete_detection_failures": sum(
                        row["error_category"] == "complete detection failure"
                        for row in primary
                    ),
                    "poor_contour_cases": sum(
                        row["error_category"] == "detected but poor contour agreement"
                        for row in primary
                    ),
                    "moderate_good_overlap_cases": sum(
                        row["error_category"] == "moderate-to-good overlap"
                        for row in primary
                    ),
                }
            )
    return case_rows, lesion_rows, summaries


def create_figures(export_dir: Path, summaries: List[dict]) -> None:
    import matplotlib.pyplot as plt

    phase_label = {
        "arterial": "Arterial",
        "portal_venous": "Portal/venous",
        "delayed": "Delayed",
        "native": "Native",
    }

    for cohort in sorted({row["cohort"] for row in summaries}):
        rows = [row for row in summaries if row["cohort"] == cohort]
        rows.sort(key=lambda row: PHASES.index(row["phase"]))
        labels = [phase_label[row["phase"]] for row in rows]
        means = np.asarray([float(row["mean_dice"]) for row in rows])
        low = means - np.asarray([float(row["bootstrap_95_ci_low"]) for row in rows])
        high = np.asarray([float(row["bootstrap_95_ci_high"]) for row in rows]) - means

        fig, ax = plt.subplots(figsize=(6.7, 3.8))
        x = np.arange(len(rows))
        ax.errorbar(x, means, yerr=np.vstack([low, high]), fmt="o", capsize=4)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean case-level Dice")
        ax.set_title(f"{cohort}: phase-wise performance")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            export_dir / f"Figure_{cohort}_Phasewise_Dice_600dpi.png",
            dpi=600,
            facecolor="white",
        )
        plt.close(fig)

        recalls = [float(row["lesion_recall"]) for row in rows]
        precision = [float(row["lesion_precision"]) for row in rows]
        f1 = [float(row["lesion_f1"]) for row in rows]
        width = 0.23
        fig, ax = plt.subplots(figsize=(6.7, 3.8))
        ax.bar(x - width, precision, width, label="Precision")
        ax.bar(x, recalls, width, label="Recall")
        ax.bar(x + width, f1, width, label="F1")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion")
        ax.set_title(f"{cohort}: phase-wise lesion detection")
        ax.legend(frameon=False, ncol=3)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            export_dir / f"Figure_{cohort}_Phasewise_Lesion_Metrics_600dpi.png",
            dpi=600,
            facecolor="white",
        )
        plt.close(fig)


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.export_dir.mkdir(parents=True, exist_ok=True)

    require(args.predict_exe, "file")
    require(args.nnunet_results, "dir")

    mapping_path = args.work_dir / "phasewise_mapping.csv"

    if args.evaluate_only:
        with mapping_path.open("r", encoding="utf-8", newline="") as handle:
            mapping_rows = list(csv.DictReader(handle))
    else:
        mapping_rows: List[dict] = []
        if "liver" in args.cohorts:
            print("[PREPARE] LiverHccSeg")
            mapping_rows.extend(prepare_liver(args))
        if "openswiss" in args.cohorts:
            print("[PREPARE] OpenSwissHCC")
            mapping_rows.extend(prepare_openswiss(args))
        write_csv(mapping_path, mapping_rows)
        write_csv(args.export_dir / "phasewise_mapping.csv", mapping_rows)

    if args.prepare_only:
        print(f"Prepared inputs and references under: {args.work_dir}")
        return 0

    if not args.evaluate_only:
        cohort_names = []
        if "liver" in args.cohorts:
            cohort_names.append("LiverHccSeg")
        if "openswiss" in args.cohorts:
            cohort_names.append("OpenSwissHCC")
        for cohort in cohort_names:
            for phase in args.phases:
                run_prediction(args, cohort, phase)

    case_rows, lesion_rows, summaries = evaluate(args, mapping_rows)
    write_csv(args.export_dir / "external_phasewise_case_metrics.csv", case_rows)
    write_csv(args.export_dir / "external_phasewise_lesion_metrics.csv", lesion_rows)
    write_csv(args.export_dir / "external_phasewise_summary.csv", summaries)

    result = {
        "analysis": "Phase-wise external MRI evaluation using unchanged Dataset801 five-fold ensemble",
        "phases": list(args.phases),
        "cohorts": list(args.cohorts),
        "model": {
            "dataset": args.dataset,
            "configuration": args.configuration,
            "trainer": args.trainer,
            "plans": args.plans,
            "folds": list(args.folds),
            "fine_tuning": False,
        },
        "openswiss_arterial_selection": (
            "For each subject, choose the water-reconstruction arterial acquisition with the "
            "largest number of available HCC phase-specific masks; ties use plain arterial, "
            "TTC-2, TTC-1, then TTC-3. Selection is independent of model output."
        ),
        "summary": summaries,
    }
    with (args.export_dir / "external_phasewise_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)

    create_figures(args.export_dir, summaries)

    lines = [
        "EXTERNAL PHASE-WISE HCC SEGMENTATION ANALYSIS",
        "=============================================",
        "",
    ]
    for row in summaries:
        lines.append(
            f"{row['cohort']} | {row['phase']} | n={row['primary_cases']} | "
            f"mean Dice {row['mean_dice']:.4f} "
            f"({row['bootstrap_95_ci_low']:.4f}-{row['bootstrap_95_ci_high']:.4f}) | "
            f"lesion precision {row['lesion_precision']:.4f} | "
            f"recall {row['lesion_recall']:.4f} | F1 {row['lesion_f1']:.4f} | "
            f"FP/case {row['false_positives_per_case']:.4f}"
        )
    (args.export_dir / "external_phasewise_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\nFinal small result package: {args.export_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
