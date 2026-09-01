# Portability and path sanitization

The public reproducibility package is a sanitized copy of the research
workspace. The original research files and model checkpoints have not
been modified.

## Python scripts

Scripts that historically contained workstation-specific absolute paths
were changed only in the release copy.

For OpenSwissHCC workflow scripts, set the local data/project root with:

    export HCC_PROJECT_ROOT=/path/to/local/OpenSwissHCC/project

If `HCC_PROJECT_ROOT` is not set, these scripts use the current working
directory as the project root.

Standard nnU-Net environment variables should be configured before
running nnU-Net operations:

    export nnUNet_raw=/path/to/nnunet/raw
    export nnUNet_preprocessed=/path/to/nnunet/preprocessed
    export nnUNet_results=/path/to/nnunet/results

## Derived output files

Some archived nnU-Net and reviewer-analysis result files originally
contained absolute local prediction/reference paths. In this public
release those path prefixes were replaced by descriptive placeholders,
including:

- `<HCC_PROJECT_ROOT>`
- `<NNUNET_ROOT>`
- `<NNUNET_PREPROCESSED>`
- `<NNUNET_RESULTS>`
- `<HCC_PHASEWISE_WORK>`
- `<OPENSWISS_EXTERNAL_ROOT>`

Case identifiers, metric values, fold assignments, hashes, diagnostic
labels and statistical results were not changed by this sanitization.

See `repository_audit/SANITIZATION_LOG.tsv` for SHA-256 hashes of each
file before and after release-copy sanitization.
