import os
from pathlib import Path
import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(os.environ.get("HCC_PROJECT_ROOT", ".")).resolve()

R = ROOT / "reviewer_reanalysis"

LESION = ROOT / "OpenSwissHCC_lesion_manifest.csv"
CASE = R / "OpenSwissHCC_case_phase_metrics.tsv"
COMP = R / "OpenSwissHCC_component_matching_caselevel.tsv"

SEED = 20260823
BOOT = 10000

PHASES = [
    "arterial",
    "portal_venous",
    "delayed",
    "native",
]

MASK_COLUMNS = {
    "arterial": "selected_arterial_mask",
    "portal_venous": "venous_mask",
    "delayed": "delayed_mask",
    "native": "native_mask",
}

LABELS = {
    "arterial": "Arterial",
    "portal_venous": "Venous",
    "delayed": "Delayed",
    "native": "Native",
}


# ============================================================
# HELPERS
# ============================================================

def present(x):
    if pd.isna(x):
        return False
    x = str(x).strip()
    return x not in ("", "nan", "None")


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan

    z = 1.959963984540054
    p = k / n

    den = 1 + z*z/n

    centre = (
        p + z*z/(2*n)
    ) / den

    half = (
        z
        * math.sqrt(
            p*(1-p)/n
            + z*z/(4*n*n)
        )
        / den
    )

    return (
        centre-half,
        centre+half
    )


def paired_bootstrap(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    d = a - b
    rng = np.random.default_rng(SEED)

    vals = np.empty(BOOT)

    for i in range(BOOT):
        idx = rng.integers(
            0, len(d), len(d)
        )
        vals[i] = d[idx].mean()

    return (
        d.mean(),
        np.quantile(vals, 0.025),
        np.quantile(vals, 0.975),
    )


def holm(ps):
    ps = np.asarray(ps, float)
    m = len(ps)

    order = np.argsort(ps)
    out = np.empty(m)

    previous = 0

    for rank, idx in enumerate(order):
        v = min(
            1.0,
            (m-rank)*ps[idx]
        )

        previous = max(
            previous,
            v
        )

        out[idx] = previous

    return out


def bootstrap_component(rows):
    rng = np.random.default_rng(SEED)
    n = len(rows)

    prec = np.zeros(BOOT)
    rec = np.zeros(BOOT)

    tp = rows["tp"].to_numpy()
    fp = rows["fp"].to_numpy()
    fn = rows["fn"].to_numpy()

    for i in range(BOOT):

        idx = rng.integers(
            0,
            n,
            n
        )

        T = int(tp[idx].sum())
        Fp = int(fp[idx].sum())
        Fn = int(fn[idx].sum())

        prec[i] = (
            T/(T+Fp)
            if T+Fp
            else 0
        )

        rec[i] = (
            T/(T+Fn)
            if T+Fn
            else 0
        )

    return (
        np.quantile(
            prec,
            [0.025, 0.975]
        ),
        np.quantile(
            rec,
            [0.025, 0.975]
        ),
    )


# ============================================================
# BUILD PHASE-SPECIFIC REFERENCE COMPLETENESS
# ============================================================

les = pd.read_csv(LESION)

les["HCC"] = pd.to_numeric(
    les["HCC"],
    errors="coerce"
)

les["Lesion_size_mm"] = pd.to_numeric(
    les["Lesion_size_mm"],
    errors="coerce"
)

hcc = les[
    les["HCC"] == 1
].copy()

print("="*80)
print("PHASE-SPECIFIC HCC REFERENCE COMPLETENESS")
print("="*80)

complete = {}

for phase in PHASES:

    col = MASK_COLUMNS[phase]

    subjects = []

    for subject, g in hcc.groupby(
        "subject_id"
    ):

        ok = g[col].map(
            present
        ).all()

        if ok:
            subjects.append(subject)

    complete[phase] = set(subjects)

    print(
        f"{LABELS[phase]:<10}: "
        f"{len(subjects)} complete subjects"
    )


# These are the counts in the audited manuscript.
expected = {
    "arterial": 59,
    "portal_venous": 57,
    "delayed": 56,
    "native": 39,
}

for phase, n in expected.items():

    if len(complete[phase]) != n:

        raise RuntimeError(
            f"{phase}: expected {n} complete "
            f"subjects, found "
            f"{len(complete[phase])}"
        )

all_four = set.intersection(
    *[
        complete[p]
        for p in PHASES
    ]
)

print(
    "\nComplete in ALL FOUR phases:",
    len(all_four)
)

if len(all_four) != 38:

    raise RuntimeError(
        f"Expected 38 all-four complete "
        f"subjects, found {len(all_four)}"
    )

print("\nPASS: completeness audit matches manuscript.")


# ============================================================
# LOAD EXISTING CASE RESULTS
# ============================================================

case = pd.read_csv(
    CASE,
    sep="\t"
)

case["reference_complete"] = [
    row.subject
    in complete[row.phase]
    for row in case.itertuples()
]

primary = case[
    case["reference_complete"]
].copy()


# ============================================================
# VERIFY ORIGINAL MANUSCRIPT PHASE RESULTS
# ============================================================

print()
print("="*80)
print("CORRECTED PHASE-SPECIFIC PRIMARY RESULTS")
print("="*80)

verification = []

for phase in PHASES:

    g = primary[
        primary["phase"] == phase
    ]

    detected = int(
        g["detected"].sum()
    )

    row = {
        "phase": phase,
        "n": len(g),
        "mean_dice": g["dice"].mean(),
        "median_dice": g["dice"].median(),
        "detected_cases": detected,
        "complete_misses":
            len(g)-detected,
    }

    verification.append(row)

    print(
        f"{LABELS[phase]:<10} "
        f"n={len(g):2d}  "
        f"mean Dice={g['dice'].mean():.4f}  "
        f"median={g['dice'].median():.4f}  "
        f"Dice>0={detected}/{len(g)}  "
        f"misses={len(g)-detected}/{len(g)}"
    )

pd.DataFrame(
    verification
).to_csv(
    R /
    "Corrected_phase_primary_verification.tsv",
    sep="\t",
    index=False
)


# ============================================================
# REVIEWER 2 / 4:
# SIZE + MULTIFOCALITY
# ============================================================

meta = (
    hcc.groupby("subject_id")
    .agg(
        hcc_count=(
            "lesion_index",
            "count"
        ),
        min_hcc_mm=(
            "Lesion_size_mm",
            "min"
        ),
        max_hcc_mm=(
            "Lesion_size_mm",
            "max"
        ),
        median_hcc_mm=(
            "Lesion_size_mm",
            "median"
        ),
    )
    .reset_index()
)

meta["multifocal"] = np.where(
    meta["hcc_count"] > 1,
    "Multifocal",
    "Single HCC"
)

meta["size20"] = np.where(
    meta["max_hcc_mm"] < 20,
    "All HCC <20 mm",
    "At least one HCC >=20 mm"
)

meta["size3"] = pd.cut(
    meta["max_hcc_mm"],
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

primary = primary.merge(
    meta,
    on="subject",
    how="left",
    suffixes=("", "_meta")
)

strata = []

for phase in PHASES:

    p = primary[
        primary["phase"] == phase
    ]

    for variable in [
        "size20",
        "size3",
        "multifocal",
    ]:

        for level, g in p.groupby(
            variable,
            observed=True
        ):

            n = len(g)
            det = int(
                g["detected"].sum()
            )

            lo, hi = wilson(
                det,
                n
            )

            strata.append({
                "phase": phase,
                "stratification":
                    variable,
                "group": str(level),
                "n": n,
                "detected": det,
                "detection_rate":
                    det/n,
                "detection_CI_low":
                    lo,
                "detection_CI_high":
                    hi,
                "mean_dice":
                    g["dice"].mean(),
                "median_dice":
                    g["dice"].median(),
            })

strata = pd.DataFrame(
    strata
)

strata.to_csv(
    R /
    "Corrected_size_multifocality_analysis.tsv",
    sep="\t",
    index=False
)

print()
print("="*80)
print("REVIEWER 2/4: <20 mm vs >=20 mm")
print("="*80)

print(
    strata[
        strata[
            "stratification"
        ] == "size20"
    ][
        [
            "phase",
            "group",
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
print("="*80)
print("REVIEWER 2/4: SINGLE vs MULTIFOCAL")
print("="*80)

print(
    strata[
        strata[
            "stratification"
        ] == "multifocal"
    ][
        [
            "phase",
            "group",
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


# ============================================================
# REVIEWER 5 Q5:
# SIX PAIRED COMPARISONS IN THE 38 ALL-FOUR-COMPLETE CASES
# ============================================================

paired = case[
    case["subject"].isin(
        all_four
    )
].copy()

wide = paired.pivot(
    index="subject",
    columns="phase",
    values="dice"
)

if len(wide) != 38:

    raise RuntimeError(
        f"Expected paired n=38; "
        f"got {len(wide)}"
    )

pairs = list(
    itertools.combinations(
        PHASES,
        2
    )
)

rows = []
rawp = []

for a, b in pairs:

    x = wide[a].to_numpy()
    y = wide[b].to_numpy()

    diff, lo, hi = (
        paired_bootstrap(
            x,
            y
        )
    )

    try:
        w = wilcoxon(
            x,
            y,
            zero_method="wilcox",
            alternative="two-sided"
        )

        stat = float(
            w.statistic
        )

        p = float(
            w.pvalue
        )

    except ValueError:

        stat = 0.0
        p = 1.0

    rawp.append(p)

    rows.append({
        "phase_A": a,
        "phase_B": b,
        "paired_n": len(x),
        "mean_A": x.mean(),
        "mean_B": y.mean(),
        "mean_difference_A_minus_B":
            diff,
        "bootstrap_CI_low": lo,
        "bootstrap_CI_high": hi,
        "wilcoxon_W": stat,
        "wilcoxon_p": p,
    })

adjusted = holm(rawp)

for row, p in zip(
    rows,
    adjusted
):
    row["holm_p"] = p

pairs_df = pd.DataFrame(
    rows
)

pairs_df.to_csv(
    R /
    "Corrected_all6_paired_phase_comparisons_n38.tsv",
    sep="\t",
    index=False
)

print()
print("="*80)
print("REVIEWER 5 Q5: ALL SIX PAIRED COMPARISONS — n=38")
print("="*80)

print(
    pairs_df.to_string(
        index=False
    )
)


# ============================================================
# REVIEWER 5 Q3:
# FILTER COMPONENT ANALYSIS TO COMPLETE REFERENCES
# ============================================================

comp = pd.read_csv(
    COMP,
    sep="\t"
)

comp["subject"] = (
    comp["case_id"]
    .str.extract(
        r"(\d{3})$"
    )[0]
    .map(
        lambda x:
        f"sub-{x}"
    )
)

comp["reference_complete"] = [
    row.subject
    in complete[row.phase]
    for row in comp.itertuples()
]

comp = comp[
    comp["reference_complete"]
].copy()

summary = []

for (
    phase,
    method,
    threshold
), g in comp.groupby(
    [
        "phase",
        "method",
        "threshold"
    ]
):

    tp = int(
        g["tp"].sum()
    )

    fp = int(
        g["fp"].sum()
    )

    fn = int(
        g["fn"].sum()
    )

    precision = (
        tp/(tp+fp)
        if tp+fp
        else 0
    )

    recall = (
        tp/(tp+fn)
        if tp+fn
        else 0
    )

    f1 = (
        2*precision*recall
        /(precision+recall)
        if precision+recall
        else 0
    )

    pci, rci = (
        bootstrap_component(g)
    )

    summary.append({
        "phase": phase,
        "method": method,
        "threshold": threshold,
        "n_cases": len(g),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "precision_CI_low":
            pci[0],
        "precision_CI_high":
            pci[1],
        "recall": recall,
        "recall_CI_low":
            rci[0],
        "recall_CI_high":
            rci[1],
        "F1": f1,
        "FP_per_case":
            fp/len(g),
    })

summary = pd.DataFrame(
    summary
)

summary.to_csv(
    R /
    "Corrected_component_matching_sensitivity.tsv",
    sep="\t",
    index=False
)

print()
print("="*80)
print("REVIEWER 5 Q3: HUNGARIAN MATCHING SENSITIVITY")
print("="*80)

print(
    summary[
        summary["method"]
        == "hungarian"
    ].to_string(
        index=False
    )
)

print()
print("="*80)
print("FILES SAVED")
print("="*80)

for f in [
    "Corrected_phase_primary_verification.tsv",
    "Corrected_size_multifocality_analysis.tsv",
    "Corrected_all6_paired_phase_comparisons_n38.tsv",
    "Corrected_component_matching_sensitivity.tsv",
]:

    print(
        R / f
    )

print()
print(
    "PASS: COMPLETE-REFERENCE reviewer "
    "reanalysis finished."
)
