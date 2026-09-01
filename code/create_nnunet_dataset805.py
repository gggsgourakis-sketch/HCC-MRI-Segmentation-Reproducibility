#!/usr/bin/env python3

import csv
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("HCC_PROJECT_ROOT", ".")).resolve()

IMAGE_ROOT = (
    ROOT
    / "derivatives"
    / "T1_images_registered_safe"
)

LABEL_ROOT = (
    ROOT
    / "derivatives"
    / "HCC_master_labels_registered_safe"
)

MANIFEST = (
    ROOT
    / "OpenSwissHCC_fixed_5fold_manifest.tsv"
)

NNUNET_RAW = Path(
    os.environ.get(
        "nnUNet_raw",
        "./nnunet",
    )
)

NNUNET_PREPROCESSED = Path(
    os.environ.get(
        "nnUNet_preprocessed",
        "./nnunet/preprocessed",
    )
)

SOURCE_SPLITS = (
    NNUNET_PREPROCESSED
    / "Dataset804_OpenSwissHCC_Venous"
    / "splits_final.json"
)

DATASET_NAME = "Dataset805_OpenSwissHCC_ArterialVenous"

TARGET = NNUNET_RAW / DATASET_NAME
IMAGES_TR = TARGET / "imagesTr"
LABELS_TR = TARGET / "labelsTr"

AUDIT_FILE = ROOT / "Dataset805_channel_selection_audit.tsv"


# ============================================================
# HELPERS
# ============================================================

def nifti_files(path: Path):
    return sorted(
        p for p in path.rglob("*")
        if p.is_file()
        and (
            p.name.lower().endswith(".nii.gz")
            or p.name.lower().endswith(".nii")
        )
    )


def canonical_subject(value):
    """
    Converts:
      sub-001 -> sub-001
      001     -> sub-001
      1       -> sub-001
      OSHCC_001 -> sub-001
    """
    s = str(value).strip()

    m = re.search(r"sub[-_]?(\d+)", s, re.I)
    if m:
        return f"sub-{int(m.group(1)):03d}"

    m = re.search(r"OSHCC[-_]?(\d+)", s, re.I)
    if m:
        return f"sub-{int(m.group(1)):03d}"

    m = re.fullmatch(r"\d+", s)
    if m:
        return f"sub-{int(s):03d}"

    raise RuntimeError(
        f"Cannot normalize subject identifier: {value!r}"
    )


def canonical_case(value):
    s = str(value).strip()

    m = re.search(r"OSHCC[-_]?(\d+)", s, re.I)
    if m:
        return f"OSHCC_{int(m.group(1)):03d}"

    m = re.search(r"sub[-_]?(\d+)", s, re.I)
    if m:
        return f"OSHCC_{int(m.group(1)):03d}"

    if s.isdigit():
        return f"OSHCC_{int(s):03d}"

    raise RuntimeError(
        f"Cannot normalize nnU-Net case identifier: {value!r}"
    )


def is_venous(path: Path):
    s = path.name.lower()

    return (
        "venous" in s
        or "portalvenous" in s
        or "portal_venous" in s
        or "portal-venous" in s
        or re.search(r"(^|[_\-.])pv([_\-.]|$)", s)
        is not None
    )


def is_arterial(path: Path):
    s = path.name.lower()

    return (
        "arterial" in s
        or re.search(r"ttc[123]", s) is not None
        or re.search(r"(^|[_\-.])art([_\-.]|$)", s)
        is not None
    )


def arterial_rank(path: Path):
    """
    Leakage-safe deterministic rule.

    Handles TTC filename variants such as:
      TTC1
      TTC-1
      TTC_1
      TTC-3

    Returns TTC1/TTC2/TTC3 when encoded in the filename;
    otherwise SINGLE.
    """
    s = path.name.lower()

    m = re.search(r"ttc[-_]?([123])", s)

    if m:
        return f"TTC{m.group(1)}"

    return "SINGLE"


def geom(img):
    return {
        "size": tuple(img.GetSize()),
        "spacing": tuple(img.GetSpacing()),
        "origin": tuple(img.GetOrigin()),
        "direction": tuple(img.GetDirection()),
    }


def close_tuple(a, b, tol=1e-4):
    if len(a) != len(b):
        return False
    return all(
        abs(float(x) - float(y)) <= tol
        for x, y in zip(a, b)
    )


def same_geometry(a, b):
    ga = geom(a)
    gb = geom(b)

    return (
        ga["size"] == gb["size"]
        and close_tuple(
            ga["spacing"],
            gb["spacing"]
        )
        and close_tuple(
            ga["origin"],
            gb["origin"]
        )
        and close_tuple(
            ga["direction"],
            gb["direction"]
        )
    )


def find_subject_dir(root: Path, subject: str):
    direct = root / subject

    if direct.exists():
        return direct

    candidates = [
        p for p in root.rglob(subject)
        if p.is_dir()
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise RuntimeError(
            f"No image directory found for {subject}"
        )

    raise RuntimeError(
        f"Multiple image directories found for "
        f"{subject}: {candidates}"
    )


def find_label(subject: str):
    direct_candidates = nifti_files(
        LABEL_ROOT / subject
    ) if (LABEL_ROOT / subject).exists() else []

    if len(direct_candidates) == 1:
        return direct_candidates[0]

    all_labels = nifti_files(LABEL_ROOT)

    candidates = [
        p for p in all_labels
        if subject.lower() in str(p).lower()
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"{subject}: expected exactly one "
            f"HCC master label, found {len(candidates)}:\n"
            + "\n".join(str(x) for x in candidates)
        )

    return candidates[0]


# ============================================================
# INPUT SANITY
# ============================================================

for required in [
    IMAGE_ROOT,
    LABEL_ROOT,
    MANIFEST,
    SOURCE_SPLITS,
]:
    if not required.exists():
        raise RuntimeError(
            f"Required input missing: {required}"
        )

if TARGET.exists():
    raise RuntimeError(
        f"\nTARGET ALREADY EXISTS:\n{TARGET}\n\n"
        "Dataset805 was NOT modified.\n"
        "Remove it manually only if you intentionally "
        "want to rebuild it."
    )


# ============================================================
# READ FIXED MANIFEST
# ============================================================

rows = []

with MANIFEST.open(
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    required_cols = {
        "nnunet_case",
        "subject",
        "HCC_positive",
        "fold",
    }

    missing = required_cols - set(
        reader.fieldnames or []
    )

    if missing:
        raise RuntimeError(
            f"Manifest missing columns: {missing}"
        )

    for r in reader:

        case = canonical_case(
            r["nnunet_case"]
        )

        subject = canonical_subject(
            r["subject"]
        )

        positive = int(
            r["HCC_positive"]
        )

        fold = int(
            r["fold"]
        )

        rows.append(
            {
                "case": case,
                "subject": subject,
                "positive": positive,
                "fold": fold,
            }
        )


if len(rows) != 132:
    raise RuntimeError(
        f"Expected 132 manifest rows, "
        f"found {len(rows)}"
    )

if len({
    r["case"] for r in rows
}) != 132:
    raise RuntimeError(
        "Duplicate nnU-Net case IDs in manifest"
    )

if len({
    r["subject"] for r in rows
}) != 132:
    raise RuntimeError(
        "Duplicate subject IDs in manifest"
    )


# ============================================================
# VERIFY EXACT DATASET804 SPLITS
# ============================================================

splits = json.loads(
    SOURCE_SPLITS.read_text()
)

if len(splits) != 5:
    raise RuntimeError(
        f"Expected five Dataset804 splits, "
        f"found {len(splits)}"
    )

all_cases = {
    r["case"] for r in rows
}

for fold_idx in range(5):

    expected_val = {
        r["case"]
        for r in rows
        if r["fold"] == fold_idx
    }

    expected_train = (
        all_cases - expected_val
    )

    actual_train = set(
        splits[fold_idx]["train"]
    )

    actual_val = set(
        splits[fold_idx]["val"]
    )

    if actual_train != expected_train:
        raise RuntimeError(
            f"Fold {fold_idx}: Dataset804 train "
            f"split does not match manifest."
        )

    if actual_val != expected_val:
        raise RuntimeError(
            f"Fold {fold_idx}: Dataset804 validation "
            f"split does not match manifest."
        )


print()
print("=" * 80)
print("DATASET804 SPLITS VERIFIED")
print("=" * 80)

for i, s in enumerate(splits):
    print(
        f"Fold {i}: "
        f"train={len(s['train'])}, "
        f"val={len(s['val'])}"
    )


# ============================================================
# CREATE TARGET
# ============================================================

IMAGES_TR.mkdir(
    parents=True,
    exist_ok=False
)

LABELS_TR.mkdir(
    parents=True,
    exist_ok=False
)


# ============================================================
# BUILD TWO-CHANNEL DATASET
# ============================================================

audit = []

selection_counts = Counter()

positive_labels = 0
negative_labels = 0

try:

    for idx, r in enumerate(
        sorted(
            rows,
            key=lambda x: x["case"]
        ),
        start=1,
    ):

        case = r["case"]
        subject = r["subject"]

        subject_dir = find_subject_dir(
            IMAGE_ROOT,
            subject
        )

        files = nifti_files(
            subject_dir
        )

        venous_candidates = [
            p for p in files
            if is_venous(p)
        ]

        arterial_candidates = [
            p for p in files
            if is_arterial(p)
        ]

        if len(venous_candidates) != 1:
            raise RuntimeError(
                f"\n{subject}: expected exactly one "
                f"registered venous image, found "
                f"{len(venous_candidates)}:\n"
                + "\n".join(
                    str(p)
                    for p in venous_candidates
                )
            )

        venous = venous_candidates[0]

        if len(arterial_candidates) == 0:
            raise RuntimeError(
                f"{subject}: no arterial "
                f"candidate found."
            )

        # ----------------------------------------------------
        # Deterministic arterial choice
        # ----------------------------------------------------

        ttc3 = [
            p for p in arterial_candidates
            if arterial_rank(p) == "TTC3"
        ]

        ttc1 = [
            p for p in arterial_candidates
            if arterial_rank(p) == "TTC1"
        ]

        ttc2 = [
            p for p in arterial_candidates
            if arterial_rank(p) == "TTC2"
        ]

        single = [
            p for p in arterial_candidates
            if arterial_rank(p) == "SINGLE"
        ]

        if ttc3:
            if len(ttc3) != 1:
                raise RuntimeError(
                    f"{subject}: ambiguous TTC3: "
                    f"{ttc3}"
                )

            # If TTC3 exists, this is the deterministic
            # triple-arterial rule.
            arterial = ttc3[0]
            rule = "TTC3"

        elif ttc1:

            if len(ttc1) != 1:
                raise RuntimeError(
                    f"{subject}: ambiguous TTC1: "
                    f"{ttc1}"
                )

            # No TTC3 exists: TTC1-only acquisition.
            # TTC2 without TTC3 would be unexpected.
            if ttc2:
                raise RuntimeError(
                    f"{subject}: TTC1/TTC2 found "
                    f"without TTC3. Refusing to guess."
                )

            arterial = ttc1[0]
            rule = "TTC1"

        elif len(single) == 1:

            arterial = single[0]
            rule = "SINGLE"

        else:
            raise RuntimeError(
                f"\n{subject}: arterial acquisition "
                f"is ambiguous.\nCandidates:\n"
                + "\n".join(
                    str(p)
                    for p in arterial_candidates
                )
            )

        selection_counts[rule] += 1

        label = find_label(
            subject
        )

        # ----------------------------------------------------
        # Geometry QA
        # ----------------------------------------------------

        art_img = sitk.ReadImage(
            str(arterial)
        )

        ven_img = sitk.ReadImage(
            str(venous)
        )

        lab_img = sitk.ReadImage(
            str(label)
        )

        if not same_geometry(
            art_img,
            ven_img
        ):
            raise RuntimeError(
                f"\nGEOMETRY FAILURE {subject}\n"
                f"Arterial: {arterial}\n"
                f"Venous:   {venous}\n"
                f"Arterial geometry: {geom(art_img)}\n"
                f"Venous geometry:   {geom(ven_img)}"
            )

        if not same_geometry(
            ven_img,
            lab_img
        ):
            raise RuntimeError(
                f"\nLABEL GEOMETRY FAILURE {subject}\n"
                f"Venous: {venous}\n"
                f"Label:  {label}\n"
                f"Venous geometry: {geom(ven_img)}\n"
                f"Label geometry:  {geom(lab_img)}"
            )

        # ----------------------------------------------------
        # Label QA
        # ----------------------------------------------------

        arr = sitk.GetArrayViewFromImage(
            lab_img
        )

        unique = set(
            int(x)
            for x in np.unique(arr)
        )

        if not unique.issubset(
            {0, 1}
        ):
            raise RuntimeError(
                f"{subject}: master label contains "
                f"unexpected values {sorted(unique)}"
            )

        foreground = int(
            np.count_nonzero(arr)
        )

        observed_positive = int(
            foreground > 0
        )

        if observed_positive != r["positive"]:
            raise RuntimeError(
                f"{subject}: HCC status mismatch. "
                f"Manifest={r['positive']}, "
                f"label_foreground={foreground}"
            )

        if observed_positive:
            positive_labels += 1
        else:
            negative_labels += 1

        # ----------------------------------------------------
        # nnU-Net channel mapping
        #
        # 0000 = arterial
        # 0001 = venous
        # ----------------------------------------------------

        art_dst = (
            IMAGES_TR
            / f"{case}_0000.nii.gz"
        )

        ven_dst = (
            IMAGES_TR
            / f"{case}_0001.nii.gz"
        )

        label_dst = (
            LABELS_TR
            / f"{case}.nii.gz"
        )

        shutil.copy2(
            arterial,
            art_dst
        )

        shutil.copy2(
            venous,
            ven_dst
        )

        shutil.copy2(
            label,
            label_dst
        )

        audit.append(
            {
                "nnunet_case": case,
                "subject": subject,
                "fold": r["fold"],
                "HCC_positive": r["positive"],
                "arterial_rule": rule,
                "arterial_source": str(arterial),
                "venous_source": str(venous),
                "label_source": str(label),
                "label_foreground_voxels": foreground,
                "size_xyz": "x".join(
                    str(x)
                    for x in art_img.GetSize()
                ),
                "spacing_xyz": "x".join(
                    f"{x:.8g}"
                    for x in art_img.GetSpacing()
                ),
            }
        )

        if (
            idx % 10 == 0
            or idx == 132
        ):
            print(
                f"Processed {idx}/132"
            )


except Exception:

    print()
    print(
        "BUILD FAILED - removing incomplete "
        "Dataset805 directory"
    )

    shutil.rmtree(
        TARGET,
        ignore_errors=True
    )

    raise


# ============================================================
# EXPECTED COHORT COMPOSITION
# ============================================================

if positive_labels != 63:
    raise RuntimeError(
        f"Expected 63 positive labels, "
        f"found {positive_labels}"
    )

if negative_labels != 69:
    raise RuntimeError(
        f"Expected 69 negative labels, "
        f"found {negative_labels}"
    )


# Expected acquisition structure previously audited:
#
# 85 subjects = triple arterial -> TTC3
# 20 subjects = TTC1 only        -> TTC1
# 27 subjects = single arterial  -> SINGLE

expected_selection = {
    "TTC3": 85,
    "TTC1": 20,
    "SINGLE": 27,
}

actual_selection = {
    k: selection_counts.get(k, 0)
    for k in expected_selection
}

if actual_selection != expected_selection:

    print()
    print("Observed arterial selection:")
    print(actual_selection)

    raise RuntimeError(
        "Arterial acquisition distribution does "
        "not match expected 85 TTC3 / "
        "20 TTC1 / 27 single. "
        "Dataset805 requires investigation."
    )


# ============================================================
# DATASET.JSON
# ============================================================

dataset_json = {
    "name": DATASET_NAME,
    "description": (
        "OpenSwissHCC registered arterial + venous "
        "T1 MRI. Arterial selection is deterministic "
        "and independent of lesion visibility: "
        "TTC3 for triple-arterial acquisitions, "
        "TTC1 for TTC1-only acquisitions, and the "
        "sole arterial acquisition otherwise. "
        "Foreground label is HCC only."
    ),
    "channel_names": {
        "0": "T1w_arterial",
        "1": "T1w_venous",
    },
    "labels": {
        "background": 0,
        "HCC": 1,
    },
    "numTraining": 132,
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO",
}

(
    TARGET
    / "dataset.json"
).write_text(
    json.dumps(
        dataset_json,
        indent=2
    )
    + "\n"
)


# ============================================================
# COPY EXACT DATASET804 SPLIT FILE
# ============================================================

shutil.copy2(
    SOURCE_SPLITS,
    TARGET / "splits_final.json"
)


# ============================================================
# PROVENANCE
# ============================================================

provenance = {
    "dataset_id": 805,
    "dataset_name": DATASET_NAME,
    "n_subjects": 132,
    "n_HCC_positive_subjects": 63,
    "n_HCC_negative_subjects": 69,
    "channels": {
        "0000": "registered arterial T1",
        "0001": "registered venous T1",
    },
    "arterial_selection_rule": {
        "triple_arterial": "TTC3",
        "TTC1_only": "TTC1",
        "single_arterial": "single arterial acquisition",
    },
    "arterial_selection_is_lesion_independent": True,
    "arterial_selection_counts": dict(
        selection_counts
    ),
    "labels": (
        "Existing HCC_master_labels_registered_safe; "
        "97 HCC lesions only. Non-HCC lesions remain "
        "background/hard negatives."
    ),
    "split_source": str(
        SOURCE_SPLITS
    ),
    "split_manifest": str(
        MANIFEST
    ),
    "splits_reused_verbatim_from_Dataset804": True,
}

(
    TARGET
    / "provenance.json"
).write_text(
    json.dumps(
        provenance,
        indent=2
    )
    + "\n"
)


# ============================================================
# CHANNEL-SELECTION AUDIT TSV
# ============================================================

fieldnames = list(
    audit[0].keys()
)

with AUDIT_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        delimiter="\t"
    )

    writer.writeheader()
    writer.writerows(audit)


# Copy audit into raw dataset too
shutil.copy2(
    AUDIT_FILE,
    TARGET
    / "Dataset805_channel_selection_audit.tsv"
)


# ============================================================
# FINAL FILE COUNTS
# ============================================================

n_art = len(
    list(
        IMAGES_TR.glob(
            "*_0000.nii.gz"
        )
    )
)

n_ven = len(
    list(
        IMAGES_TR.glob(
            "*_0001.nii.gz"
        )
    )
)

n_lab = len(
    list(
        LABELS_TR.glob(
            "*.nii.gz"
        )
    )
)

if (
    n_art != 132
    or n_ven != 132
    or n_lab != 132
):
    raise RuntimeError(
        f"Final file count failure: "
        f"arterial={n_art}, "
        f"venous={n_ven}, "
        f"labels={n_lab}"
    )


print()
print("=" * 80)
print("DATASET805 BUILD COMPLETE")
print("=" * 80)

print("Dataset:")
print(TARGET)

print()
print("Channels:")
print("  0000 = arterial")
print("  0001 = venous")

print()
print("Files:")
print(f"  arterial images = {n_art}")
print(f"  venous images   = {n_ven}")
print(f"  labels          = {n_lab}")

print()
print("HCC status:")
print(
    f"  positive subjects = "
    f"{positive_labels}"
)
print(
    f"  negative subjects = "
    f"{negative_labels}"
)

print()
print("Arterial selection:")
for k in [
    "TTC3",
    "TTC1",
    "SINGLE",
]:
    print(
        f"  {k:<6} = "
        f"{selection_counts[k]}"
    )

print()
print("Fixed folds:")
for i, s in enumerate(splits):
    print(
        f"  Fold {i}: "
        f"train={len(s['train'])}, "
        f"val={len(s['val'])}"
    )

print()
print(
    "GEOMETRY QA: PASS"
)
print(
    "LABEL QA: PASS"
)
print(
    "FIXED-SPLIT QA: PASS"
)
print(
    "ARTERIAL-SELECTION QA: PASS"
)
print()
print(
    "Dataset805 raw construction: PASS"
)
