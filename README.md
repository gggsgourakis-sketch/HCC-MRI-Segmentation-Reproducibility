# HCC MRI Segmentation Reproducibility Package v1.0

## Associated manuscript

**Hepatocellular Carcinoma Segmentation on MRI: Five-Fold ATLAS Benchmarking, Phase-Wise External Validation, and Exploratory MRI-to-CT Stress Testing**

This release contains code, nnU-Net configurations, patient-level fold assignments, derived manifests, evaluation outputs, reviewer-requested reanalyses, checkpoint provenance, software-environment information and the trained checkpoints required to reproduce the principal reported analyses.

## Raw medical images

Raw ATLAS, LiverHccSeg, OpenSwissHCC and HCC-TACE-Seg medical images are **not redistributed** in this release.

Source medical images must be obtained from the original public repositories under their respective licences and access conditions.

## Dataset801_HCCMRI_ATLAS

- 60 ATLAS MRI cases.
- Single-channel T1-weighted MRI.
- Binary tumour-versus-background target.
- Five patient-level folds.
- 48 training and 12 validation subjects per fold.
- Every subject contributes exactly once to the OOF validation result.
- nnU-Net v2 3D full-resolution configuration.
- Original validation used the terminal model state after training.
- The five `checkpoint_final.pth` files are provided in the corresponding checkpoint archive.
- Mean patient-level OOF Dice: 0.5315.
- Dataset-aggregated pooled-voxel Dice: 0.6231.

## Dataset804_OpenSwissHCC_Venous

- 132 OpenSwissHCC subjects.
- 63 HCC-positive subjects.
- 69 HCC-negative subjects.
- Single venous T1 input.
- Fixed five-fold patient-level cross-validation.
- Primary analysis uses the prespecified best-checkpoint rule.
- This experiment is a within-OpenSwissHCC development/ablation analysis and is not external validation.

## Dataset805_OpenSwissHCC_ArterialVenous

- Same 132 subjects as Dataset804.
- Identical five patient-level folds.
- Two input channels: arterial and venous T1.
- Primary analysis uses the frozen best/pre-collapse-best checkpoints listed in `checkpoints/PRIMARY_CHECKPOINT_MANIFEST.tsv`.
- This experiment is a within-OpenSwissHCC development/ablation analysis and is not external validation.

## Dataset804 versus Dataset805 paired OOF result

Among 63 HCC-positive subjects:

- venous-only mean Dice: 0.2269;
- arterial+venous mean Dice: 0.3705;
- paired mean difference: +0.1435;
- bootstrap 95% CI: +0.0520 to +0.2343;
- Wilcoxon signed-rank p = 0.00223.

Any-overlap HCC detection increased from 52.4% to 68.3%; exact McNemar p = 0.0639.

Among 69 HCC-negative subjects:

- correct-empty specificity increased from 30.4% to 42.0%;
- exact McNemar p = 0.2005;
- mean false-positive voxel burden per negative scan decreased from 640.3 to 209.8;
- Wilcoxon p = 0.0394.

## External OpenSwissHCC evaluation

The unchanged Dataset801 ATLAS ensemble was separately applied to OpenSwissHCC arterial, venous, delayed and native T1 inputs.

All 132 subjects underwent inference in every phase. Positive-case reference completeness varies by phase because some released phase-specific HCC masks are incomplete. Tumour-positive segmentation performance and tumour-negative false-positive safety are therefore reported separately.

True-negative examinations are not assigned Dice = 1 and pooled into the positive-case mean Dice.

## Reviewer 5 Q4: ATLAS comparator reconciliation

The present ATLAS result is not directly interchangeable with the Karabağ et al. benchmark.

The current analysis uses:

- binary tumour-versus-background segmentation;
- all 60 ATLAS subjects exactly once as OOF validation cases;
- five patient-level folds of 48 training / 12 validation subjects;
- nnU-Net v2.6.4;
- Z-score normalisation;
- target spacing approximately 3.0 × 1.042 × 1.042 mm;
- patch size 48 × 192 × 224;
- batch size 2.

Karabağ et al. report tumour Dice of 0.915 ± 0.016 for five-fold validation and 0.892 for their independent held-out test set.

Their exact patient-level fold assignments and complete generated nnU-Net run artefacts are not available sufficiently to reconstruct their numerical experiment unambiguously. The associated reviewer audit therefore provides a protocol-level reconciliation rather than claiming numerical equivalence.

## Directory structure

- `code/` — reconstruction, registration, dataset-building and evaluation code.
- `configuration/` — nnU-Net dataset definitions, fingerprints, plans and patient-level folds.
- `manifests/` — derived subject, lesion and fold manifests.
- `results/` — primary OOF metrics and within-cohort ablation results.
- `reviewer_reanalysis/` — reviewer-requested reanalysis outputs.
- `repository_audit/` — release inventories and reproducibility audit files.
- `environment/` — software versions and frozen Python package environment.
- `checkpoints/` — primary checkpoint provenance manifest.

## Large checkpoint archives

The primary trained checkpoints are supplied separately as:

1. `Dataset801_ATLAS_final_checkpoints.tar`
2. `Dataset804_venous_primary_checkpoints.tar`
3. `Dataset805_arterial_venous_primary_checkpoints.tar`

See `checkpoints/PRIMARY_CHECKPOINT_MANIFEST.tsv` for archive-member names, analysis roles and SHA-256 hashes.

## Methodological safeguards

No probability threshold was selected using final OOF performance.

Checkpoint-selection rules were defined according to the corresponding experiment and were not retrospectively changed because another checkpoint produced a numerically preferable result.

Dataset804/805 results must not be described as external validation.

Matched-lesion Dice is conditional on successful spatial localisation and should be interpreted alongside lesion detection, complete misses and false-positive burden.

## Permanent identifiers

Zenodo DOI: **[TO BE INSERTED BEFORE PUBLICATION]**

GitHub repository: **https://github.com/gggsgourakis-sketch/HCC-MRI-Segmentation-Reproducibility**

## Licensing

This is a mixed-license release.

- Original project software under `code/`: MIT.
- Author-generated documentation and derived numerical outputs: CC BY 4.0.
- Dataset801 ATLAS-derived checkpoint archive: CC BY-NC-SA 4.0.
- Dataset804/805 OpenSwissHCC-derived checkpoint archives: CC BY 4.0.
- Raw medical imaging is not redistributed.

See `LICENSE` and `LICENSES.md` for details.
