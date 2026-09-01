import os
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from statistics import median
import math
import re
import SimpleITK as sitk

ROOT = Path(os.environ.get("HCC_PROJECT_ROOT", ".")).resolve()

SOURCE = ROOT / "derivatives" / "manual_lesion_annotations"
REGISTERED = (
    ROOT / "derivatives" / "T1_lesion_masks_registered_safe_test"
)

SUBJECTS = ["sub-001", "sub-006"]


def binary(path):
    img = sitk.ReadImage(str(path))
    return sitk.Cast(img > 0, sitk.sitkUInt8)


def voxel_count(img):
    stats = sitk.StatisticsImageFilter()
    stats.Execute(img)
    return int(stats.GetSum())


def voxel_volume(img):
    v = 1.0
    for x in img.GetSpacing():
        v *= x
    return v


def physical_volume(img):
    return voxel_count(img) * voxel_volume(img)


def centroid(img):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(img)

    if not stats.HasLabel(1):
        return None

    return stats.GetCentroid(1)


def centroid_distance(a, b):
    ca = centroid(a)
    cb = centroid(b)

    if ca is None or cb is None:
        return float("nan")

    return math.sqrt(
        sum((x-y)**2 for x, y in zip(ca, cb))
    )


def dice(a, b):
    overlap = sitk.LabelOverlapMeasuresImageFilter()
    overlap.Execute(a, b)
    return overlap.GetDiceCoefficient()


def lesion_number(name):
    m = re.search(r"-L([1-5])_seg\.nii\.gz$", name)
    return int(m.group(1)) if m else None


def phase(name):
    if "arterial-TTC-1" in name:
        return "arterial-TTC1"
    if "arterial-TTC-2" in name:
        return "arterial-TTC2"
    if "arterial-TTC-3" in name:
        return "arterial-TTC3"
    if "phase-arterial" in name:
        return "arterial"
    if "phase-native" in name:
        return "native"
    if "phase-venous" in name:
        return "venous"
    if "phase-delayed" in name:
        return "delayed"
    return "unknown"


all_dice = []

print("=" * 86)
print("OpenSwissHCC REGISTRATION QUANTITATIVE QA")
print("=" * 86)

for sid in SUBJECTS:

    print("\n" + "=" * 86)
    print(sid)
    print("=" * 86)

    regdir = REGISTERED / sid / "dyn"
    srcdir = SOURCE / sid / "dyn"

    registered_files = sorted(regdir.glob("*.nii.gz"))

    if not registered_files:
        raise RuntimeError(
            f"No registered masks found for {sid}"
        )

    # --------------------------------------------------------
    # Physical volume preservation
    # --------------------------------------------------------

    print("\nPHYSICAL VOLUME PRESERVATION")
    print("-" * 86)

    groups = defaultdict(list)

    for regfile in registered_files:

        srcfile = srcdir / regfile.name

        if not srcfile.exists():
            raise RuntimeError(
                f"Missing source mask for {regfile.name}"
            )

        src = binary(srcfile)
        reg = binary(regfile)

        sv = physical_volume(src)
        rv = physical_volume(reg)

        change = 100.0 * (rv - sv) / sv

        lesion = lesion_number(regfile.name)
        ph = phase(regfile.name)

        groups[lesion].append((ph, reg))

        flag = ""

        # Heuristic flag only; not a formal failure criterion.
        if abs(change) > 25:
            flag = "  <-- REVIEW"

        print(
            f"L{lesion} {ph:16s} "
            f"source={sv:9.1f} mm3   "
            f"registered={rv:9.1f} mm3   "
            f"change={change:+7.1f}%"
            f"{flag}"
        )

    # --------------------------------------------------------
    # Cross-phase overlap
    # --------------------------------------------------------

    print("\nCROSS-PHASE OVERLAP AFTER REGISTRATION")
    print("-" * 86)

    for lesion in sorted(groups):

        items = groups[lesion]

        print(f"\nL{lesion}")

        if len(items) < 2:
            print("  Only one T1 mask available.")
            continue

        for (phase1, img1), (phase2, img2) in combinations(items, 2):

            d = dice(img1, img2)
            c = centroid_distance(img1, img2)

            all_dice.append(d)

            flag = ""

            # These are screening flags, not hard acceptance criteria.
            if d < 0.30 or c > 10:
                flag = "  <-- REVIEW"

            print(
                f"  {phase1:16s} vs {phase2:16s} "
                f"Dice={d:.3f}   "
                f"centroid distance={c:.2f} mm"
                f"{flag}"
            )


print("\n" + "=" * 86)
print("OVERALL")
print("=" * 86)

if all_dice:
    print("Cross-phase comparisons:", len(all_dice))
    print("Median registered Dice:", round(median(all_dice), 3))
    print("Minimum registered Dice:", round(min(all_dice), 3))
    print("Maximum registered Dice:", round(max(all_dice), 3))

print("\nQA COMPLETE")
