# Changes from supplied files

The supplied implementation was inspected directly. The original earlier research notebook was not supplied, so equivalence to an unseen historical implementation is not asserted.

## Scientific configuration corrections

- `Config.max_area`: 4200 → 5500. The supplied 4200 value violated its own validation because the optimizer upper bound was 5500. This aligns configuration with the requested baseline and the existing CLI/notebook area setting.
- CLI `--filter-mode`: fixed → optimize; CLI `--optimizer-count-mode`: contours → legacy_original.
- Notebook configuration: fixed → optimize, with explicit legacy_original objective. Manual count is read from the supplied filename/count metadata and passed into `filter_contours`.

No processing function bodies for spectral features, connectivity, K-means, morphology, contour metrics, optimization, gallery matching, classification, or final counting were changed. In particular, the legacy objective's cumulative four-individual contribution for a contour above 3.5× median area and the noise/uncertain counting policy are preserved. Compatibility modes remain available but are not the documented workflow.

## Portability and documentation

- Renamed supplied `(1)` script/notebook filenames to canonical names.
- Default gallery path now points to the root-level `tuta_gallery/`.
- Removed notebook personal absolute paths; examples and metadata use relative paths.
- Replaced stale fixed-mode introductory comments and notebook explanations.
- Restored the notebook inline plotting backend after module import; explicitly imported `display`.
- Added notebook optimization-history display, configuration/history exports, and a compact JSON summary for comparison, without rerunning its segmentation.
- Added JupyterLab, nbconvert and ipykernel dependencies; retained pytest and added five regression tests. No existing test script was supplied.
- Added README, candidate-column definitions, metadata, validation report, exact tested dependency lock, gitignore, and this change record.
- Retained all 209 gallery PNGs and the original candidates CSV unchanged.
- Retained three real example filenames and RGB pixel values/resolutions; stripped PNG ancillary metadata by lossless re-encoding. An incomplete generated copy of the third photograph was detected in validation, recreated from the intact supplied ZIP, and rechecked against its original pixels.
- Copied both supplied illustrative images to the documented relative paths.
- Added LICENSE as a pending-permissions notice, not an open-source grant. No authors, funding identifiers, publications or DOI were invented.

The supplied 17-count comparison figure agrees with the verified final count on that photograph. Disagreement on other photographs is reported rather than tuned away through methodological changes.
