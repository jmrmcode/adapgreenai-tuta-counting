# Validation report

Validated on 2026-11-02 with Linux and Python 3.12.14. See `requirements-lock.txt` for exact dependency versions.

## Results

| Photograph | Observed | Internal objective | Retained contours | Touching additions | Final | Difference |
|---|---:|---:|---:|---:|---:|---:|
| photo_01_05_2026_12-01-01cut.png | 17 | 17 | 17 | 0 | 17 | 0.00% |
| photo_20_05_2026_12-01-01cut.png | 201 | 161 | 158 | 3 | 161 | 19.90% |
| photo_21_04_2026_17-01-01cut.png | 66 | 65 | 65 | 1 | 66 | 0.00% |

**These differences are calibration diagnostics, not independent predictive accuracy.**

The 201-count photograph remains outside the 5% target: the search evaluated 9,759 candidates and retained the baseline thresholds, giving 161 under both the legacy objective and final gallery rule. No processing changes were made to force a match. The 66-count photograph accepted the baseline immediately because its internal objective of 65 was already within tolerance; gallery correction then gave 66.

## Checks performed

- Created a clean virtual environment without system site packages; installed every dependency from requirements.txt successfully. `pip check` reported no broken requirements.
- All imports succeeded; all five pytest regression checks passed. These cover configuration/CLI defaults, the cumulative legacy area correction, observed-count-driven threshold changes, noise/uncertain/pair counting, and the complete galleries.
- Ran all three real photographs at original resolution through the CLI with default optimization; verified all six PNGs and CSV/JSON outputs, and equality between summed contour contributions and final counts.
- Executed every notebook code cell in order in a newly created in-process IPython kernel for each image, using the same isolated Python environment. The full counting summary and selected thresholds matched the CLI exactly for all three photographs.
- Standard separate-process Jupyter kernel startup could not be validated in this execution environment: TCP and IPC socket binding both returned Operation not permitted. The fresh in-process kernel checks validate cell ordering, computations, plotting calls, and exports, but do not substitute for a browser-based JupyterLab startup check on the collaborator’s machine.
- Kernel shutdown emitted a pending IOPub cleanup-task warning after successful execution. It did not interrupt any cell or the count comparisons.
- All 209 reference PNGs loaded: 159 single, 18 touching, 32 noise. Every candidate metadata record maps to an existing class/filename, and crop-coordinate spans match its PNG dimensions.
- Example RGB arrays were checked pixel-for-pixel against the supplied ZIP. Resolution and pixels are unchanged; rewritten PNGs contain no EXIF/GPS metadata.
- README local links resolve; the example Python API and CLI use actual implemented arguments. Personal absolute paths are absent from committed source/notebook files.

## Selected thresholds

| Observed | Minimum area | Maximum area | Minimum solidity | Optimizer evaluations |
|---:|---:|---:|---:|---:|
| 17 | 791.945360 | 4649.775073 | 0.430000 | 9627 |
| 201 | 600.000000 | 5500.000000 | 0.430000 | 9759 |
| 66 | 600.000000 | 5500.000000 | 0.430000 | 1 |

Reproduce software regression checks with `python -m pytest -q`. Reproduce the analyses using the README batch example or the notebook image selector. The manual counts are user-supplied and were not independently recounted. Equivalence to an earlier historical implementation cannot be established without that implementation; the legacy objective is verified against the formula in the supplied script.

## Pending before public release

Confirm rights-holder and author attribution, code license, and permissions for photographs, galleries, figures and candidate data. Supply verified funding acknowledgement if desired. Run the normal JupyterLab fresh-kernel workflow locally. Consider further scientific evaluation of the high-capture discrepancy without interpreting these count-guided results as independent validation.
