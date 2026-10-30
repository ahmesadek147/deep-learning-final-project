# CUB EDA and Data Audit

Source: `Multimodal-Fatima/CUB_train:train` (5994 images, 200 classes).

## Findings

- Derived deterministic split sizes used by training: 4796 train / 1198 validation.
- Untouched test metadata: 5794 images.
- Class counts range from 29 to 30; no classes are missing.
- Corrupted/missing images detected: **0**.
- Exact duplicate pixel hashes detected: **1** (PNG-normalized SHA-256).
- Image modes: `{'RGB': 5990, 'L': 4}`; width range 121–500, height range 120–500.
- Caption typo candidates saved: 506; these are not applied to source images/labels.

## Corrected versus unchanged

- Corrected only in derived caption artifacts: text normalization, conservative typo candidates, duplicate-caption filtering.
- Intentionally unchanged: source dataset files, labels, class mapping, and all images; no samples were removed.

## Outputs

See `outputs/eda/` for dimensions, class distribution, caption charts, and representative samples. See `outputs/data_audit/` for broken-image, duplicate, channel, caption-cleaning, and findings JSON files.
