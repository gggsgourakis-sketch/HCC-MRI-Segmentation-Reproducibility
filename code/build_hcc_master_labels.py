import os
from pathlib import Path
import csv
import re
import SimpleITK as sitk

ROOT = Path(os.environ.get("HCC_PROJECT_ROOT", ".")).resolve()

PARTICIPANTS = ROOT / "participants.tsv"

MRI_ROOT = (
    ROOT / "derivatives" / "T1_images_registered_safe"
)

MASK_ROOT = (
    ROOT / "derivatives" / "T1_lesion_masks_registered_safe"
)

OUT_ROOT = (
    ROOT / "derivatives" / "HCC_master_labels_registered_safe"
)

AUDIT_FILE = (
    ROOT / "HCC_master_label_selection_audit.tsv"
)


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def detect_column(headers, aliases):
    lookup = {norm(h): h for h in headers}

    for alias in aliases:
        n = norm(alias)
        if n in lookup:
            return lookup[n]

    raise RuntimeError(
        f"Could not identify column from aliases {aliases}.\n"
        f"Available columns:\n{headers}"
    )


def is_hcc(v):
    v = str(v).strip().lower()

    return v in {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
        "hcc",
    }


def lesion_number(v):
    s = str(v).strip()

    if not s or s.lower() in {
        "na", "nan", "none", ""
    }:
        return None

    try:
        return int(float(s))
    except Exception:
        m = re.search(r"([1-5])", s)
        return int(m.group(1)) if m else None


def same_geometry(a, b, tol=1e-5):
    if a.GetSize() != b.GetSize():
        return False

    for x, y in zip(a.GetSpacing(), b.GetSpacing()):
        if abs(x-y) > tol:
            return False

    for x, y in zip(a.GetOrigin(), b.GetOrigin()):
        if abs(x-y) > tol:
            return False

    for x, y in zip(a.GetDirection(), b.GetDirection()):
        if abs(x-y) > tol:
            return False

    return True


def foreground(img):
    f = sitk.StatisticsImageFilter()
    f.Execute(sitk.Cast(img > 0, sitk.sitkUInt8))
    return int(f.GetSum())


def get_phase(filename):
    if "phase-venous" in filename:
        return "venous"

    if "phase-arterial" in filename:
        return "arterial"

    if "phase-delayed" in filename:
        return "delayed"

    if "phase-native" in filename:
        return "native"

    return None


# ============================================================
# Read metadata
# ============================================================

with PARTICIPANTS.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f, delimiter="\t")

    headers = reader.fieldnames

    if not headers:
        raise RuntimeError("participants.tsv has no header.")

    subject_col = detect_column(
        headers,
        [
            "ID",
            "participant_id",
            "participant id",
            "subject",
            "subject_id",
        ]
    )

    lesion_col = detect_column(
        headers,
        [
            "Lesion index",
            "lesion_index",
            "lesion index",
        ]
    )

    hcc_col = detect_column(
        headers,
        [
            "HCC",
            "hcc",
        ]
    )

    rows = list(reader)


print("Detected columns:")
print(" subject:", subject_col)
print(" lesion: ", lesion_col)
print(" HCC:    ", hcc_col)


# ============================================================
# HCC subject-lesion pairs
# ============================================================

hcc_pairs = set()

for row in rows:

    sid = str(row[subject_col]).strip()
    lesion = lesion_number(row[lesion_col])

    if lesion is None:
        continue

    if is_hcc(row[hcc_col]):
        hcc_pairs.add((sid, lesion))


print()
print("HCC subject-lesion pairs:", len(hcc_pairs))

if len(hcc_pairs) != 97:
    raise RuntimeError(
        f"Expected 97 HCC lesions, found {len(hcc_pairs)}. "
        "Stop and inspect metadata interpretation."
    )


# ============================================================
# Build one HCC label per subject
# ============================================================

OUT_ROOT.mkdir(parents=True, exist_ok=True)

audit = []

positive_subjects = 0
negative_subjects = 0
selected_lesions = 0

phase_counts = {
    "venous": 0,
    "arterial": 0,
    "delayed": 0,
    "native": 0,
}


for i in range(1, 133):

    sid = f"sub-{i:03d}"

    mri_dir = MRI_ROOT / sid / "dyn"
    mask_dir = MASK_ROOT / sid / "dyn"

    venous_files = sorted(
        mri_dir.glob("*phase-venous*_T1w.nii.gz")
    )

    if len(venous_files) != 1:
        raise RuntimeError(
            f"{sid}: expected exactly one registered venous MRI, "
            f"found {len(venous_files)}"
        )

    ref = sitk.ReadImage(str(venous_files[0]))

    master = sitk.Image(
        ref.GetSize(),
        sitk.sitkUInt8
    )

    master.CopyInformation(ref)

    subject_pairs = sorted(
        lesion
        for s, lesion in hcc_pairs
        if s == sid
    )

    if subject_pairs:
        positive_subjects += 1
    else:
        negative_subjects += 1

    for lesion in subject_pairs:

        pattern = f"*-L{lesion}_seg.nii.gz"

        candidates = sorted(
            mask_dir.glob(pattern)
        )

        if not candidates:
            raise RuntimeError(
                f"{sid} L{lesion}: no registered T1 mask."
            )

        by_phase = {
            "venous": [],
            "arterial": [],
            "delayed": [],
            "native": [],
        }

        for p in candidates:
            ph = get_phase(p.name)

            if ph:
                by_phase[ph].append(p)

        chosen = None
        chosen_phase = None

        for ph in [
            "venous",
            "arterial",
            "delayed",
            "native",
        ]:

            files = by_phase[ph]

            if len(files) == 0:
                continue

            if len(files) > 1:
                raise RuntimeError(
                    f"{sid} L{lesion}: multiple {ph} "
                    f"masks found:\n"
                    + "\n".join(str(x) for x in files)
                )

            chosen = files[0]
            chosen_phase = ph
            break

        if chosen is None:
            raise RuntimeError(
                f"{sid} L{lesion}: no usable dynamic T1 mask."
            )

        label = sitk.ReadImage(str(chosen))
        label = sitk.Cast(label > 0, sitk.sitkUInt8)

        if not same_geometry(ref, label):
            raise RuntimeError(
                f"{sid} L{lesion}: geometry mismatch "
                f"for {chosen.name}"
            )

        nvox = foreground(label)

        if nvox == 0:
            raise RuntimeError(
                f"{sid} L{lesion}: selected label is empty."
            )

        master = sitk.Or(master, label)

        selected_lesions += 1
        phase_counts[chosen_phase] += 1

        audit.append({
            "subject": sid,
            "lesion": f"L{lesion}",
            "selected_phase": chosen_phase,
            "selected_mask": chosen.name,
            "foreground_voxels": nvox,
        })

    out_dir = OUT_ROOT / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    outfile = (
        out_dir /
        f"{sid}_HCC_seg.nii.gz"
    )

    sitk.WriteImage(master, str(outfile))


# ============================================================
# Audit TSV
# ============================================================

with AUDIT_FILE.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    fields = [
        "subject",
        "lesion",
        "selected_phase",
        "selected_mask",
        "foreground_voxels",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
        delimiter="\t"
    )

    writer.writeheader()
    writer.writerows(audit)


# ============================================================
# Final validation
# ============================================================

label_files = sorted(
    OUT_ROOT.glob("sub-*/sub-*_HCC_seg.nii.gz")
)

nonempty = 0
empty = 0

for p in label_files:

    x = sitk.ReadImage(str(p))

    n = foreground(x)

    if n > 0:
        nonempty += 1
    else:
        empty += 1


print()
print("=" * 72)
print("HCC MASTER LABEL SUMMARY")
print("=" * 72)

print("Labels written:", len(label_files))
print("HCC lesions selected:", selected_lesions)
print("Positive subjects:", positive_subjects)
print("Negative subjects:", negative_subjects)
print("Non-empty master labels:", nonempty)
print("Empty master labels:", empty)

print()
print("Selected source phases:")
for k, v in phase_counts.items():
    print(f"  {k:10s}: {v}")

print()
print("Audit:")
print(AUDIT_FILE)

if len(label_files) != 132:
    raise RuntimeError("Expected 132 master labels.")

if selected_lesions != 97:
    raise RuntimeError("Expected 97 HCC lesions.")

if nonempty != 63:
    raise RuntimeError(
        f"Expected 63 HCC-positive subjects, got {nonempty}."
    )

if empty != 69:
    raise RuntimeError(
        f"Expected 69 negative subjects, got {empty}."
    )

print()
print("MASTER LABEL QA: PASS")
