from pathlib import Path
import csv
import json
import os
import random
import shutil
from datetime import datetime

ROOT = Path(os.environ.get("HCC_PROJECT_ROOT", ".")).resolve()

MRI_ROOT = (
    ROOT / "derivatives" / "T1_images_registered_safe"
)

LABEL_ROOT = (
    ROOT / "derivatives" / "HCC_master_labels_registered_safe"
)

PARTICIPANTS = ROOT / "participants.tsv"

raw_env = os.environ.get("nnUNet_raw")

if not raw_env:
    raise RuntimeError(
        "nnUNet_raw is not set. Activate/configure nnU-Net first."
    )

NNUNET_RAW = Path(raw_env)

DATASET_NAME = "Dataset804_OpenSwissHCC_Venous"
DATASET = NNUNET_RAW / DATASET_NAME

if DATASET.exists():
    raise RuntimeError(
        f"{DATASET} already exists.\n"
        "Nothing has been overwritten."
    )

imagesTr = DATASET / "imagesTr"
labelsTr = DATASET / "labelsTr"

imagesTr.mkdir(parents=True)
labelsTr.mkdir(parents=True)


# ============================================================
# Determine HCC-positive subjects
# ============================================================

positive_subjects = set()

with PARTICIPANTS.open(
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:

        sid = row["ID"].strip()

        if row["HCC"].strip() == "1":
            positive_subjects.add(sid)


if len(positive_subjects) != 63:
    raise RuntimeError(
        f"Expected 63 HCC-positive subjects, "
        f"found {len(positive_subjects)}"
    )


# ============================================================
# Copy 132 venous MRI volumes and corresponding master labels
# ============================================================

case_map = {}
all_cases = []

for i in range(1, 133):

    sid = f"sub-{i:03d}"
    case = f"OSHCC_{i:03d}"

    mri_dir = MRI_ROOT / sid / "dyn"

    venous = sorted(
        mri_dir.glob("*phase-venous*_T1w.nii.gz")
    )

    if len(venous) != 1:
        raise RuntimeError(
            f"{sid}: expected exactly one venous image, "
            f"found {len(venous)}"
        )

    label = (
        LABEL_ROOT /
        sid /
        f"{sid}_HCC_seg.nii.gz"
    )

    if not label.exists():
        raise FileNotFoundError(label)

    dst_img = imagesTr / f"{case}_0000.nii.gz"
    dst_lab = labelsTr / f"{case}.nii.gz"

    shutil.copy2(venous[0], dst_img)
    shutil.copy2(label, dst_lab)

    case_map[sid] = case
    all_cases.append(case)


# ============================================================
# nnU-Net dataset.json
# ============================================================

dataset_json = {
    "channel_names": {
        "0": "T1w_venous"
    },
    "labels": {
        "background": 0,
        "HCC": 1
    },
    "numTraining": 132,
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO"
}

with (DATASET / "dataset.json").open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dataset_json,
        f,
        indent=4
    )


# ============================================================
# Fixed patient-level 5-fold split
#
# Positive distribution:
#   13, 13, 13, 12, 12 = 63
#
# Negative distribution:
#   13, 14, 14, 14, 14 = 69
#
# This exact split will later be reused for every phase ablation.
# ============================================================

positive_cases = [
    case_map[sid]
    for sid in sorted(positive_subjects)
]

negative_cases = [
    case_map[f"sub-{i:03d}"]
    for i in range(1, 133)
    if f"sub-{i:03d}" not in positive_subjects
]

if len(positive_cases) != 63:
    raise RuntimeError("Positive case count != 63.")

if len(negative_cases) != 69:
    raise RuntimeError("Negative case count != 69.")


rng_pos = random.Random(20260812)
rng_neg = random.Random(20260813)

rng_pos.shuffle(positive_cases)
rng_neg.shuffle(negative_cases)

positive_per_fold = [13, 13, 13, 12, 12]
negative_per_fold = [13, 14, 14, 14, 14]


def split_by_counts(items, counts):

    result = []
    start = 0

    for n in counts:

        result.append(
            items[start:start+n]
        )

        start += n

    if start != len(items):
        raise RuntimeError(
            "Fold split accounting error."
        )

    return result


pos_folds = split_by_counts(
    positive_cases,
    positive_per_fold
)

neg_folds = split_by_counts(
    negative_cases,
    negative_per_fold
)

case_set = set(all_cases)

splits = []

for fold in range(5):

    val = sorted(
        pos_folds[fold] +
        neg_folds[fold]
    )

    train = sorted(
        case_set - set(val)
    )

    splits.append({
        "train": train,
        "val": val
    })


# ============================================================
# Write fixed nnU-Net split
# ============================================================

with (DATASET / "splits_final.json").open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        splits,
        f,
        indent=4
    )


# ============================================================
# Permanent independent fold manifest
# ============================================================

manifest = (
    ROOT /
    "OpenSwissHCC_fixed_5fold_manifest.tsv"
)

case_to_fold = {}

for fold, split in enumerate(splits):

    for case in split["val"]:
        case_to_fold[case] = fold


with manifest.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(
        f,
        delimiter="\t"
    )

    writer.writerow([
        "subject",
        "nnunet_case",
        "HCC_positive",
        "fold"
    ])

    for i in range(1, 133):

        sid = f"sub-{i:03d}"
        case = case_map[sid]

        writer.writerow([
            sid,
            case,
            int(sid in positive_subjects),
            case_to_fold[case]
        ])


# ============================================================
# Provenance
# ============================================================

provenance = {
    "dataset": DATASET_NAME,
    "created": datetime.now().isoformat(),
    "source_dataset": "OpenSwissHCC",
    "subjects": 132,
    "HCC_positive_subjects": 63,
    "HCC_negative_subjects": 69,
    "HCC_lesions": 97,
    "input_channel": "registered venous T1-weighted MRI",
    "foreground_definition": "HCC only",
    "non_HCC_lesions": (
        "retained on MRI but treated as background/hard negatives"
    ),
    "master_label_source": str(LABEL_ROOT),
    "registered_MRI_source": str(MRI_ROOT),
    "master_label_audit": str(
        ROOT / "HCC_master_label_selection_audit.tsv"
    ),
    "fold_manifest": str(manifest),
    "master_label_phase_selection": {
        "venous": 89,
        "arterial_fallback": 8,
        "delayed": 0,
        "native": 0
    }
}

with (DATASET / "provenance.json").open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        provenance,
        f,
        indent=4
    )


# ============================================================
# Final QA
# ============================================================

n_images = len(
    list(imagesTr.glob("*_0000.nii.gz"))
)

n_labels = len(
    list(labelsTr.glob("*.nii.gz"))
)

print("=" * 76)
print(DATASET_NAME)
print("=" * 76)

print("Images:", n_images)
print("Labels:", n_labels)
print(
    "HCC-positive patients:",
    len(positive_cases)
)
print(
    "HCC-negative patients:",
    len(negative_cases)
)

print()

seen_val = []

for fold in range(5):

    val = splits[fold]["val"]
    train = splits[fold]["train"]

    pos = sum(
        c in positive_cases
        for c in val
    )

    neg = len(val) - pos

    seen_val.extend(val)

    print(
        f"Fold {fold}: "
        f"train={len(train)}, "
        f"val={len(val)}, "
        f"HCC+={pos}, "
        f"HCC-={neg}"
    )


if n_images != 132:
    raise RuntimeError(
        f"Expected 132 images, found {n_images}"
    )

if n_labels != 132:
    raise RuntimeError(
        f"Expected 132 labels, found {n_labels}"
    )

if len(seen_val) != 132:
    raise RuntimeError(
        "Validation fold total is not 132."
    )

if len(set(seen_val)) != 132:
    raise RuntimeError(
        "A patient occurs in validation more than once."
    )

if set(seen_val) != case_set:
    raise RuntimeError(
        "Validation folds do not cover all patients."
    )


print()
print("Dataset:")
print(DATASET)

print()
print("Fold manifest:")
print(manifest)

print()
print("DATASET804 BUILD QA: PASS")
