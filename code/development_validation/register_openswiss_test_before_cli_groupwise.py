import os
from pathlib import Path
import shutil
import SimpleITK as sitk

ROOT = Path(os.environ.get("HCC_PROJECT_ROOT", ".")).resolve()

IMG_ROOT = ROOT / "images"
MASK_ROOT = ROOT / "derivatives" / "manual_lesion_annotations"
REG_ROOT = ROOT / "derivatives" / "T1_registration_transforms"

IMG_OUT = ROOT / "derivatives" / "T1_images_registered_safe_test"
MASK_OUT = ROOT / "derivatives" / "T1_lesion_masks_registered_safe_test"

SUBJECTS = ["sub-001", "sub-006"]


def prepare(pm, order):
    # Distributed _1 files refer to old filenames such as
    # ./TransformParameters.0.txt. We explicitly compose the stages instead.
    pm["InitialTransformParametersFileName"] = ("NoInitialTransform",)
    pm["FinalBSplineInterpolationOrder"] = (str(order),)
    return pm


def read_pair(regdir, stem, order):
    f0 = regdir / f"pairwise_registration_transform_parameters_0_{stem}.txt"
    f1 = regdir / f"pairwise_registration_transform_parameters_1_{stem}.txt"

    if not f0.exists():
        raise FileNotFoundError(f0)
    if not f1.exists():
        raise FileNotFoundError(f1)

    p0 = sitk.ReadParameterFile(str(f0))
    p1 = sitk.ReadParameterFile(str(f1))

    return [prepare(p0, order), prepare(p1, order)]


def transform(image, maps):
    if not maps:
        return image

    tx = sitk.TransformixImageFilter()
    tx.LogToConsoleOff()
    tx.SetMovingImage(image)
    tx.SetTransformParameterMap(maps[0])

    for pm in maps[1:]:
        tx.AddTransformParameterMap(pm)

    tx.Execute()
    return tx.GetResultImage()


def groupwise(image, group_file, position, nphases, order):
    vec = sitk.VectorOfImage()

    for _ in range(nphases):
        vec.push_back(image)

    image4d = sitk.JoinSeries(vec)

    pm = sitk.ReadParameterFile(str(group_file))
    pm = prepare(pm, order)

    tx = sitk.TransformixImageFilter()
    tx.LogToConsoleOff()
    tx.SetMovingImage(image4d)
    tx.SetTransformParameterMap(pm)
    tx.Execute()

    result4d = tx.GetResultImage()

    return result4d[:, :, :, position]


def chains(regdir, six, order):
    if six:
        n_a1 = read_pair(regdir, "native_to_arterial_TTC_1", order)
        a1_a2 = read_pair(regdir, "arterial_TTC_1_to_arterial_TTC_2", order)
        a2_a3 = read_pair(regdir, "arterial_TTC_2_to_arterial_TTC_3", order)
        a3_v  = read_pair(regdir, "arterial_TTC_3_to_venous", order)
        d_v   = read_pair(regdir, "delayed_to_venous", order)

        return {
            "native": n_a1 + a1_a2 + a2_a3 + a3_v,
            "art1": a1_a2 + a2_a3 + a3_v,
            "art2": a2_a3 + a3_v,
            "art3": a3_v,
            "venous": [],
            "delayed": d_v,
        }

    n_a = read_pair(regdir, "native_to_arterial", order)
    a_v = read_pair(regdir, "arterial_to_venous", order)
    d_v = read_pair(regdir, "delayed_to_venous", order)

    return {
        "native": n_a + a_v,
        "arterial": a_v,
        "venous": [],
        "delayed": d_v,
    }


def phase(name, six):
    if "phase-native" in name:
        return "native"
    if "phase-venous" in name:
        return "venous"
    if "phase-delayed" in name:
        return "delayed"

    if six:
        if "arterial-TTC-1" in name:
            return "art1"
        if "arterial-TTC-2" in name:
            return "art2"
        if "arterial-TTC-3" in name:
            return "art3"
    else:
        if "phase-arterial" in name:
            return "arterial"

    return None


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


def voxel_count(mask):
    s = sitk.StatisticsImageFilter()
    s.Execute(sitk.Cast(mask > 0, sitk.sitkUInt8))
    return int(s.GetSum())


def run_subject(sid):
    print("\n" + "="*78)
    print(sid)
    print("="*78)

    imgdir = IMG_ROOT / sid / "dyn"
    maskdir = MASK_ROOT / sid / "dyn"
    regdir = REG_ROOT / sid / "dyn"

    npair = len(list(regdir.glob(
        "pairwise_registration_transform_parameters_*.txt"
    )))

    if npair == 10:
        six = True
        nphases = 6
        positions = {
            "native": 0,
            "art1": 1,
            "art2": 2,
            "art3": 3,
            "venous": 4,
            "delayed": 5,
        }
    elif npair == 6:
        six = False
        nphases = 4
        positions = {
            "native": 0,
            "arterial": 1,
            "venous": 2,
            "delayed": 3,
        }
    else:
        raise RuntimeError(f"{sid}: unexpected transform count {npair}")

    print("Protocol:", "6-phase" if six else "4-phase")

    group_file = regdir / "groupwise_registration_transform_parameters_0.txt"

    iout = IMG_OUT / sid / "dyn"
    mout = MASK_OUT / sid / "dyn"

    for d in (iout, mout):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # MRI
    # B-spline interpolation order 3
    # ========================================================

    c = chains(regdir, six, 3)

    images = {}

    for p in sorted(imgdir.glob("*acq-water*_T1w.nii.gz")):
        ph = phase(p.name, six)
        if ph:
            images[ph] = p

    missing = set(positions) - set(images)

    if missing:
        raise RuntimeError(
            f"{sid}: missing MRI phases: {sorted(missing)}"
        )

    registered = {}

    print("\nMRI")

    for ph in positions:
        src = sitk.ReadImage(str(images[ph]), sitk.sitkFloat32)

        x = transform(src, c[ph])

        x = groupwise(
            x,
            group_file,
            positions[ph],
            nphases,
            3
        )

        sitk.WriteImage(x, str(iout / images[ph].name))
        registered[ph] = x

        print(
            f"{ph:9s}",
            "source", src.GetSize(),
            "->", x.GetSize(),
            "spacing",
            tuple(round(v,5) for v in x.GetSpacing())
        )

    ref = registered["venous"]

    for ph, x in registered.items():
        if not same_geometry(ref, x):
            raise RuntimeError(
                f"{sid}: MRI geometry mismatch for {ph}"
            )

    print("MRI COMMON GEOMETRY: PASS")

    # ========================================================
    # MASKS
    # nearest-neighbour / interpolation order 0
    # ========================================================

    if not maskdir.exists():
        print("No T1 masks.")
        return

    c_mask = chains(regdir, six, 0)

    n_masks = 0

    print("\nMASKS")

    for p in sorted(maskdir.glob("*.nii.gz")):
        ph = phase(p.name, six)

        if ph is None:
            continue

        src = sitk.ReadImage(str(p))
        src = sitk.Cast(src > 0, sitk.sitkUInt8)

        before = voxel_count(src)

        if before == 0:
            raise RuntimeError(f"Empty source mask: {p}")

        x = transform(
            sitk.Cast(src, sitk.sitkFloat32),
            c_mask[ph]
        )

        x = groupwise(
            x,
            group_file,
            positions[ph],
            nphases,
            0
        )

        x = sitk.Cast(x > 0.5, sitk.sitkUInt8)

        after = voxel_count(x)

        if after == 0:
            raise RuntimeError(
                f"Mask became empty: {p.name}"
            )

        if not same_geometry(ref, x):
            raise RuntimeError(
                f"{sid}: mask geometry mismatch: {p.name}"
            )

        sitk.WriteImage(x, str(mout / p.name))

        print(
            p.name,
            f": {before} -> {after} voxels"
        )

        n_masks += 1

    print("Masks processed:", n_masks)
    print("MASK GEOMETRY/BINARY QA: PASS")


print("="*78)
print("OpenSwissHCC SAFE REGISTRATION TEST")
print("="*78)

print(sitk.Version())
print(
    "Transformix:",
    hasattr(sitk, "TransformixImageFilter")
)

for sid in SUBJECTS:
    run_subject(sid)

print("\n" + "="*78)
print("ALL TEST SUBJECTS COMPLETED SUCCESSFULLY")
print("="*78)
