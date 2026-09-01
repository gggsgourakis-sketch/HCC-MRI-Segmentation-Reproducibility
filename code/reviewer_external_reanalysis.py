import os
from pathlib import Path
import itertools
import math
import warnings

import numpy as np
import pandas as pd
import nibabel as nib

from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(os.environ.get("HCC_PROJECT_ROOT", ".")).resolve()

WORK = Path(
    "./hcc_phasewise_external_work/OpenSwissHCC"
)

LESION_MANIFEST = ROOT / "OpenSwissHCC_lesion_manifest.csv"
SUBJECT_MANIFEST = ROOT / "OpenSwissHCC_subject_manifest.csv"

OUT = ROOT / "reviewer_reanalysis"
OUT.mkdir(parents=True, exist_ok=True)

PHASES = (
    "arterial",
    "portal_venous",
    "delayed",
    "native",
)

PHASE_LABEL = {
    "arterial": "Arterial",
    "portal_venous": "Venous",
    "delayed": "Delayed",
    "native": "Native",
}

BOOT = 10000
SEED = 20260823

CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)

IOU_THRESHOLDS = (
    ("any_overlap", 0.0),
    ("iou_0.01", 0.01),
    ("iou_0.05", 0.05),
    ("iou_0.10", 0.10),
)


# ============================================================
# HELPERS
# ============================================================

def load_mask(path):
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj) > 0


def dice_score(gt, pred):
    ng = int(gt.sum())
    npred = int(pred.sum())

    if ng + npred == 0:
        return 1.0

    inter = int(np.logical_and(gt, pred).sum())
    return 2.0 * inter / (ng + npred)


def iou_score(gt, pred):
    union = int(np.logical_or(gt, pred).sum())

    if union == 0:
        return 1.0

    return int(np.logical_and(gt, pred).sum()) / union


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)

    p = k / n

    denom = 1 + z*z/n

    centre = (
        p + z*z/(2*n)
    ) / denom

    half = (
        z
        * math.sqrt(
            p*(1-p)/n
            + z*z/(4*n*n)
        )
        / denom
    )

    return centre - half, centre + half


def bootstrap_mean_diff(a, b, reps=BOOT, seed=SEED):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) != len(b):
        raise RuntimeError("Paired arrays differ in length")

    d = a - b
    rng = np.random.default_rng(seed)

    means = np.empty(reps, dtype=float)

    for i in range(reps):
        idx = rng.integers(0, len(d), len(d))
        means[i] = d[idx].mean()

    return (
        d.mean(),
        np.quantile(means, 0.025),
        np.quantile(means, 0.975),
    )


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)

    order = np.argsort(pvalues)

    adjusted = np.empty(m, dtype=float)

    running = 0.0

    for rank, idx in enumerate(order):
        value = min(
            1.0,
            (m - rank) * pvalues[idx]
        )

        running = max(running, value)
        adjusted[idx] = running

    return adjusted


# ============================================================
# COMPONENT ANALYSIS
# ============================================================

def component_tables(gt, pred):
    gt_lab, ng = ndimage.label(
        gt,
        structure=CONNECTIVITY
    )

    pred_lab, npred = ndimage.label(
        pred,
        structure=CONNECTIVITY
    )

    gt_sizes = np.bincount(
        gt_lab.ravel(),
        minlength=ng + 1
    )[1:]

    pred_sizes = np.bincount(
        pred_lab.ravel(),
        minlength=npred + 1
    )[1:]

    overlap = np.zeros(
        (ng, npred),
        dtype=np.int64
    )

    if ng > 0 and npred > 0:

        mask = (gt_lab > 0) & (pred_lab > 0)

        g = gt_lab[mask] - 1
        p = pred_lab[mask] - 1

        code = g * npred + p

        counts = np.bincount(
            code,
            minlength=ng * npred
        )

        overlap = counts.reshape(
            ng,
            npred
        )

    iou = np.zeros_like(
        overlap,
        dtype=float
    )

    if ng > 0 and npred > 0:

        denom = (
            gt_sizes[:, None]
            + pred_sizes[None, :]
            - overlap
        )

        valid = denom > 0
        iou[valid] = (
            overlap[valid]
            / denom[valid]
        )

    return (
        ng,
        npred,
        overlap,
        iou,
        gt_sizes,
        pred_sizes,
    )


def threshold_valid(overlap, iou, threshold):
    if threshold == 0:
        return overlap > 0

    return iou >= threshold


def greedy_matching(
    overlap,
    iou,
    threshold
):
    ng, npred = overlap.shape

    if ng == 0 or npred == 0:
        return []

    valid = threshold_valid(
        overlap,
        iou,
        threshold
    )

    candidates = []

    for i in range(ng):
        for j in range(npred):

            if valid[i, j]:

                candidates.append(
                    (
                        int(overlap[i, j]),
                        float(iou[i, j]),
                        i,
                        j,
                    )
                )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1]
        ),
        reverse=True
    )

    used_gt = set()
    used_pred = set()
    matches = []

    for ov, iv, i, j in candidates:

        if i in used_gt:
            continue

        if j in used_pred:
            continue

        used_gt.add(i)
        used_pred.add(j)

        matches.append(
            (i, j, ov, iv)
        )

    return matches


def optimal_matching(
    overlap,
    iou,
    threshold
):
    ng, npred = overlap.shape

    if ng == 0 or npred == 0:
        return []

    valid = threshold_valid(
        overlap,
        iou,
        threshold
    )

    # Lexicographic objective:
    # first maximise number of valid matches,
    # then maximise total IoU.
    score = (
        valid.astype(float) * 1000.0
        + iou
    )

    rows, cols = linear_sum_assignment(
        score,
        maximize=True
    )

    matches = []

    for i, j in zip(rows, cols):

        if valid[i, j]:

            matches.append(
                (
                    int(i),
                    int(j),
                    int(overlap[i, j]),
                    float(iou[i, j]),
                )
            )

    return matches


def case_component_metrics(
    gt,
    pred,
    threshold,
    method
):
    (
        ng,
        npred,
        overlap,
        iou,
        gt_sizes,
        pred_sizes,
    ) = component_tables(gt, pred)

    if method == "greedy":
        matches = greedy_matching(
            overlap,
            iou,
            threshold
        )

    elif method == "hungarian":
        matches = optimal_matching(
            overlap,
            iou,
            threshold
        )

    else:
        raise ValueError(method)

    tp = len(matches)
    fp = npred - tp
    fn = ng - tp

    matched_dice = []

    for i, j, ov, iv in matches:

        denom = (
            gt_sizes[i]
            + pred_sizes[j]
        )

        if denom > 0:
            matched_dice.append(
                2 * ov / denom
            )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_components": ng,
        "pred_components": npred,
        "matched_dice": matched_dice,
    }


def bootstrap_component_ci(
    case_rows,
    reps=BOOT,
    seed=SEED
):
    rng = np.random.default_rng(seed)

    n = len(case_rows)

    precision = np.empty(
        reps,
        dtype=float
    )

    recall = np.empty(
        reps,
        dtype=float
    )

    for r in range(reps):

        idx = rng.integers(
            0,
            n,
            n
        )

        tp = sum(
            case_rows[i]["tp"]
            for i in idx
        )

        fp = sum(
            case_rows[i]["fp"]
            for i in idx
        )

        fn = sum(
            case_rows[i]["fn"]
            for i in idx
        )

        precision[r] = (
            tp / (tp + fp)
            if tp + fp
            else 0
        )

        recall[r] = (
            tp / (tp + fn)
            if tp + fn
            else 0
        )

    return (
        np.quantile(
            precision,
            [0.025, 0.975]
        ),
        np.quantile(
            recall,
            [0.025, 0.975]
        ),
    )


# ============================================================
# LOAD METADATA
# ============================================================

lesions = pd.read_csv(
    LESION_MANIFEST
)

subjects = pd.read_csv(
    SUBJECT_MANIFEST
)

lesions["HCC_num"] = pd.to_numeric(
    lesions["HCC"],
    errors="coerce"
)

lesions["size_mm"] = pd.to_numeric(
    lesions["Lesion_size_mm"],
    errors="coerce"
)

hcc = lesions[
    lesions["HCC_num"] == 1
].copy()

print("=" * 78)
print("METADATA QA")
print("=" * 78)

print(
    "HCC lesion rows:",
    len(hcc)
)

print(
    "HCC-positive subjects:",
    hcc["subject_id"].nunique()
)

print(
    "HCC lesions missing size:",
    hcc["size_mm"].isna().sum()
)

subject_meta = (
    hcc.groupby("subject_id")
    .agg(
        hcc_count=("lesion_index", "count"),
        min_hcc_mm=("size_mm", "min"),
        median_hcc_mm=("size_mm", "median"),
        max_hcc_mm=("size_mm", "max"),
    )
    .reset_index()
)

subject_meta["multifocal"] = (
    subject_meta["hcc_count"] > 1
).astype(int)

subject_meta["multifocal_label"] = np.where(
    subject_meta["multifocal"] == 1,
    "Multifocal",
    "Single HCC"
)

subject_meta["max_size_bin"] = pd.cut(
    subject_meta["max_hcc_mm"],
    bins=[
        -np.inf,
        10,
        20,
        np.inf
    ],
    right=False,
    labels=[
        "<10 mm",
        "10-<20 mm",
        ">=20 mm"
    ]
)

subject_meta["size20_group"] = np.where(
    subject_meta["max_hcc_mm"] < 20,
    "All HCC <20 mm",
    "At least one HCC >=20 mm"
)

subject_meta["max_size_per10mm"] = (
    subject_meta["max_hcc_mm"]
    / 10.0
)

meta_lookup = (
    subject_meta
    .set_index("subject_id")
    .to_dict("index")
)


# ============================================================
# VERIFY PHASE CASE SETS
# ============================================================

phase_cases = {}

for phase in PHASES:

    pdir = (
        WORK
        / phase
        / "predictions"
    )

    ldir = (
        WORK
        / phase
        / "labelsTs"
    )

    pred = {
        p.name
        for p in pdir.glob(
            "OpenSwissHCC_*.nii.gz"
        )
    }

    labels = {
        p.name
        for p in ldir.glob(
            "OpenSwissHCC_*.nii.gz"
        )
    }

    if pred != labels:
        raise RuntimeError(
            f"{phase}: prediction/label "
            f"case sets differ"
        )

    phase_cases[phase] = pred

    print(
        phase,
        "cases:",
        len(pred)
    )

case_sets = list(
    phase_cases.values()
)

if not all(
    s == case_sets[0]
    for s in case_sets[1:]
):
    raise RuntimeError(
        "Phase case sets are not identical"
    )

common_cases = sorted(
    case_sets[0]
)

print(
    "Common complete-reference cases:",
    len(common_cases)
)

if len(common_cases) != 59:

    warnings.warn(
        f"Expected old complete set n=59, "
        f"found {len(common_cases)}"
    )


# ============================================================
# CASE-LEVEL PHASE METRICS
# ============================================================

case_rows = []

mask_cache = {}

for phase in PHASES:

    for filename in common_cases:

        case_id = filename.replace(
            ".nii.gz",
            ""
        )

        number = case_id.split("_")[-1]

        subject = f"sub-{number}"

        if subject not in meta_lookup:
            raise RuntimeError(
                f"No HCC metadata for {subject}"
            )

        gt_path = (
            WORK
            / phase
            / "labelsTs"
            / filename
        )

        pred_path = (
            WORK
            / phase
            / "predictions"
            / filename
        )

        gt = load_mask(gt_path)
        pred = load_mask(pred_path)

        if gt.shape != pred.shape:
            raise RuntimeError(
                f"{phase} {case_id}: "
                f"shape mismatch "
                f"{gt.shape} vs {pred.shape}"
            )

        inter = int(
            np.logical_and(
                gt,
                pred
            ).sum()
        )

        m = meta_lookup[subject]

        row = {
            "subject": subject,
            "case_id": case_id,
            "phase": phase,
            "phase_label":
                PHASE_LABEL[phase],
            "gt_voxels":
                int(gt.sum()),
            "pred_voxels":
                int(pred.sum()),
            "intersection_voxels":
                inter,
            "detected":
                int(inter > 0),
            "dice":
                dice_score(gt, pred),
            "iou":
                iou_score(gt, pred),
            **m,
        }

        case_rows.append(row)

        mask_cache[
            (phase, case_id)
        ] = (
            gt,
            pred
        )

case_df = pd.DataFrame(
    case_rows
)

case_df.to_csv(
    OUT
    / "OpenSwissHCC_case_phase_metrics.tsv",
    sep="\t",
    index=False
)


# ============================================================
# ANALYSIS A:
# REVIEWERS 2 / 4
# SIZE + MULTIFOCALITY
# ============================================================

stratified_rows = []

for phase in PHASES:

    p = case_df[
        case_df["phase"] == phase
    ]

    stratifications = [
        (
            "max_size_3group",
            "max_size_bin"
        ),
        (
            "size_20mm",
            "size20_group"
        ),
        (
            "multifocality",
            "multifocal_label"
        ),
    ]

    for strat_name, column in stratifications:

        for level, g in p.groupby(
            column,
            observed=False
        ):

            if len(g) == 0:
                continue

            detected = int(
                g["detected"].sum()
            )

            n = len(g)

            ci_low, ci_high = (
                wilson_ci(
                    detected,
                    n
                )
            )

            stratified_rows.append({
                "phase": phase,
                "phase_label":
                    PHASE_LABEL[phase],
                "stratification":
                    strat_name,
                "level": str(level),
                "n": n,
                "detected": detected,
                "detection_rate":
                    detected / n,
                "detection_ci_low":
                    ci_low,
                "detection_ci_high":
                    ci_high,
                "mean_dice":
                    g["dice"].mean(),
                "median_dice":
                    g["dice"].median(),
                "mean_max_hcc_mm":
                    g["max_hcc_mm"].mean(),
            })

stratified_df = pd.DataFrame(
    stratified_rows
)

stratified_df.to_csv(
    OUT
    / "OpenSwissHCC_size_multifocal_summary.tsv",
    sep="\t",
    index=False
)


# ============================================================
# ADJUSTED DETECTION MODEL
# phase + HCC size + multifocality
# repeated observations clustered by subject
# ============================================================

gee_output = []

try:
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable

    model_df = case_df.copy()

    formula = (
        "detected ~ "
        "C(phase, Treatment(reference='portal_venous')) "
        "+ max_size_per10mm "
        "+ multifocal"
    )

    model = GEE.from_formula(
        formula,
        groups="subject",
        data=model_df,
        family=Binomial(),
        cov_struct=Exchangeable()
    )

    fit = model.fit()

    ci = fit.conf_int()

    for term in fit.params.index:

        beta = float(
            fit.params[term]
        )

        low = float(
            ci.loc[term, 0]
        )

        high = float(
            ci.loc[term, 1]
        )

        gee_output.append({
            "term": term,
            "beta": beta,
            "OR": math.exp(beta),
            "CI_low_OR":
                math.exp(low),
            "CI_high_OR":
                math.exp(high),
            "p":
                float(
                    fit.pvalues[term]
                ),
        })

    pd.DataFrame(
        gee_output
    ).to_csv(
        OUT
        / "OpenSwissHCC_adjusted_detection_GEE.tsv",
        sep="\t",
        index=False
    )

    GEE_STATUS = "SUCCESS"

except Exception as e:

    GEE_STATUS = (
        "NOT RUN: "
        + repr(e)
    )


# ============================================================
# ANALYSIS C:
# REVIEWER 5 Q5
# ALL SIX PAIRED PHASE COMPARISONS
# ============================================================

wide = case_df.pivot(
    index="subject",
    columns="phase",
    values="dice"
)

pairs = list(
    itertools.combinations(
        PHASES,
        2
    )
)

pair_rows = []

raw_p = []

for a, b in pairs:

    x = wide[a].to_numpy()
    y = wide[b].to_numpy()

    diff, ci_low, ci_high = (
        bootstrap_mean_diff(
            x,
            y
        )
    )

    try:
        stat = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox"
        )

        p = float(stat.pvalue)
        w = float(stat.statistic)

    except ValueError:
        p = 1.0
        w = 0.0

    raw_p.append(p)

    pair_rows.append({
        "phase_A": a,
        "phase_B": b,
        "paired_n": len(x),
        "mean_dice_A":
            float(x.mean()),
        "mean_dice_B":
            float(y.mean()),
        "mean_difference_A_minus_B":
            diff,
        "bootstrap_CI_low":
            ci_low,
        "bootstrap_CI_high":
            ci_high,
        "wilcoxon_W":
            w,
        "wilcoxon_p":
            p,
    })

adj = holm_adjust(
    raw_p
)

for row, p_adj in zip(
    pair_rows,
    adj
):
    row["holm_p"] = float(
        p_adj
    )

pair_df = pd.DataFrame(
    pair_rows
)

pair_df.to_csv(
    OUT
    / "OpenSwissHCC_phase_pairwise_comparisons.tsv",
    sep="\t",
    index=False
)


# ============================================================
# ANALYSIS B:
# REVIEWER 5 Q3
# GREEDY VS HUNGARIAN
# + IoU THRESHOLD SENSITIVITY
# ============================================================

component_case_rows = []
component_summary_rows = []

for phase in PHASES:

    for method in (
        "greedy",
        "hungarian"
    ):

        for threshold_name, threshold in IOU_THRESHOLDS:

            local_rows = []

            for filename in common_cases:

                case_id = filename.replace(
                    ".nii.gz",
                    ""
                )

                gt, pred = mask_cache[
                    (phase, case_id)
                ]

                m = case_component_metrics(
                    gt,
                    pred,
                    threshold,
                    method
                )

                local = {
                    "phase": phase,
                    "case_id": case_id,
                    "method": method,
                    "threshold":
                        threshold_name,
                    "threshold_numeric":
                        threshold,
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                    "gt_components":
                        m["gt_components"],
                    "pred_components":
                        m["pred_components"],
                    "matched_dice_mean":
                        (
                            np.mean(
                                m[
                                    "matched_dice"
                                ]
                            )
                            if m[
                                "matched_dice"
                            ]
                            else np.nan
                        ),
                }

                local_rows.append(
                    local
                )

                component_case_rows.append(
                    local
                )

            tp = sum(
                r["tp"]
                for r in local_rows
            )

            fp = sum(
                r["fp"]
                for r in local_rows
            )

            fn = sum(
                r["fn"]
                for r in local_rows
            )

            precision = (
                tp / (tp + fp)
                if tp + fp
                else 0
            )

            recall = (
                tp / (tp + fn)
                if tp + fn
                else 0
            )

            f1 = (
                2 * precision * recall
                / (precision + recall)
                if precision + recall
                else 0
            )

            pci, rci = (
                bootstrap_component_ci(
                    local_rows
                )
            )

            matched = [
                r[
                    "matched_dice_mean"
                ]
                for r in local_rows
                if not np.isnan(
                    r[
                        "matched_dice_mean"
                    ]
                )
            ]

            component_summary_rows.append({
                "phase": phase,
                "method": method,
                "threshold":
                    threshold_name,
                "threshold_numeric":
                    threshold,
                "cases":
                    len(local_rows),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision":
                    precision,
                "precision_CI_low":
                    pci[0],
                "precision_CI_high":
                    pci[1],
                "recall":
                    recall,
                "recall_CI_low":
                    rci[0],
                "recall_CI_high":
                    rci[1],
                "F1": f1,
                "FP_per_case":
                    fp / len(
                        local_rows
                    ),
                "mean_detected_case_matched_dice":
                    (
                        np.mean(matched)
                        if matched
                        else np.nan
                    ),
            })

component_case_df = pd.DataFrame(
    component_case_rows
)

component_summary_df = pd.DataFrame(
    component_summary_rows
)

component_case_df.to_csv(
    OUT
    / "OpenSwissHCC_component_matching_caselevel.tsv",
    sep="\t",
    index=False
)

component_summary_df.to_csv(
    OUT
    / "OpenSwissHCC_component_matching_sensitivity.tsv",
    sep="\t",
    index=False
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print()
print("=" * 78)
print("REVIEWER REANALYSIS COMPLETE")
print("=" * 78)

print()
print("59-case complete-reference external cohort")
print("--------------------------------------------")

for phase in PHASES:

    g = case_df[
        case_df["phase"] == phase
    ]

    detected = int(
        g["detected"].sum()
    )

    print(
        f"{PHASE_LABEL[phase]:<10} "
        f"mean Dice={g['dice'].mean():.4f}  "
        f"median={g['dice'].median():.4f}  "
        f"detected={detected}/{len(g)} "
        f"({100*detected/len(g):.1f}%)"
    )


print()
print("=" * 78)
print("SIZE <20 mm vs >=20 mm")
print("=" * 78)

x = stratified_df[
    stratified_df[
        "stratification"
    ] == "size_20mm"
]

print(
    x[
        [
            "phase_label",
            "level",
            "n",
            "detected",
            "detection_rate",
            "mean_dice",
            "median_dice",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 78)
print("SINGLE vs MULTIFOCAL")
print("=" * 78)

x = stratified_df[
    stratified_df[
        "stratification"
    ] == "multifocality"
]

print(
    x[
        [
            "phase_label",
            "level",
            "n",
            "detected",
            "detection_rate",
            "mean_dice",
            "median_dice",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 78)
print("ADJUSTED GEE DETECTION MODEL")
print("=" * 78)

print(
    "Status:",
    GEE_STATUS
)

if gee_output:

    print(
        pd.DataFrame(
            gee_output
        ).to_string(
            index=False
        )
    )


print()
print("=" * 78)
print("ALL SIX PAIRED PHASE COMPARISONS")
print("=" * 78)

print(
    pair_df[
        [
            "phase_A",
            "phase_B",
            "paired_n",
            "mean_difference_A_minus_B",
            "bootstrap_CI_low",
            "bootstrap_CI_high",
            "wilcoxon_p",
            "holm_p",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 78)
print("MATCHING SENSITIVITY")
print("Hungarian optimal one-to-one")
print("=" * 78)

print(
    component_summary_df[
        component_summary_df[
            "method"
        ] == "hungarian"
    ][
        [
            "phase",
            "threshold",
            "TP",
            "FP",
            "FN",
            "precision",
            "precision_CI_low",
            "precision_CI_high",
            "recall",
            "recall_CI_low",
            "recall_CI_high",
            "F1",
            "FP_per_case",
        ]
    ].to_string(
        index=False
    )
)


print()
print("Files written to:")
print(OUT)

for p in sorted(
    OUT.glob(
        "OpenSwissHCC_*.tsv"
    )
):
    print("  ", p.name)

print()
print(
    "PASS: reviewer analyses completed."
)
