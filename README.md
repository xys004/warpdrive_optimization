# warp_optimization

`warp_optimization` is a TensorFlow-based constrained-optimization and post-processing workflow for the revised warp-drive energy-condition study. The repository packages the optimizer, plot generation, and output verification steps used to reproduce the paper's principal-stress diagnostics for uniformly translated, zero-vorticity single-shell and double-shell seed configurations.

The active workflow is a **parametric optimization over an analytic seed ansatz with physics-informed penalty terms**. It is not a PINN. The referee-facing pipeline is the revised script set in the repository root; historical notebook exports placed under `data/` should be treated as legacy audit material rather than as the authoritative figure-production path.

The current codebase centers on seven key files/scripts:

- `physics_core.py`: shared principal-stress eigenvalue helpers used by both the optimizer and plotter.
- `einstein_optimizer.py`: runs one optimization snapshot and exports CSV/JSON artifacts.
- `run_batch.py`: launches the fixed four-run batch used during development.
- `postprocess_plots.py`: rebuilds field maps and optimization figures directly from exported run artifacts.
- `verify_outputs.py`: checks run-bundle and plot consistency.
- `generate_run_bundle.py`: convenience wrapper that runs optimization, plotting, and verification end-to-end.
- `colab_smoke_test.ipynb`: Google Colab notebook for a lightweight cloud smoke test.

## Repository Layout

```text
warp_optimization/
|-- physics_core.py
|-- einstein_optimizer.py
|-- run_batch.py
|-- postprocess_plots.py
|-- verify_outputs.py
|-- generate_run_bundle.py
|-- colab_smoke_test.ipynb
|-- requirements.txt
|-- LICENSE
|-- CITATION.cff
`-- README.md
```

Generated outputs are written next to the run base name, for example:

```text
<base>_final_params.csv
<base>_parameters.csv
<base>_losses.csv
<base>_success_rates.csv
<base>_metadata.json
<base>_XY_rho.png
<base>_XY_WECmin.png
...
```

## Installation

The scripts were tested with Python 3.9 and TensorFlow 2.19 on Windows 10. A minimal environment can be created with:

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

If you are using Conda instead, create a Python 3.9 environment and install the same requirements.

## Run in Google Colab

This repository is published at:

```text
https://github.com/xys004/warp_bubble_optimization.git
```

The repository includes two Colab notebooks:

- `colab_smoke_test.ipynb`: reduced smoke test for installation and bundle verification
- `colab_paper_run.ipynb`: heavier single-run workflow with GPU check, bundle verification, and manuscript-parameter comparison

Recommended workflow:

1. Open `colab_smoke_test.ipynb` from the GitHub repository in Google Colab.
2. Run the smoke test to confirm the environment, TensorFlow install, and plotting pipeline.
3. Then open `colab_paper_run.ipynb` for a longer run with paper-style settings.

Because the repository is public, both notebooks can clone it directly without embedding a GitHub token.

If `verify_outputs.py` reports missing `*_final_params.csv`, `*_parameters.csv`, `*_losses.csv`, `*_success_rates.csv`, or `*_metadata.json`, that usually means the generation step failed or the runtime was reset before the bundle was written. The updated Colab notebooks now stop immediately when `generate_run_bundle.py` fails and print the files that were actually created.

## Run a Single Optimization

To run one optimization snapshot and export the raw run bundle:

```bash
python einstein_optimizer.py --domain 2 --v 0.1 --t 30.0 --seed 47 --overwrite
```

This writes files with a base name of the form:

```text
domain_<domain>_v_<velocity tag>_t_<time tag>
```

For example:

```text
domain_2_v_0p1_t_30p0
```

By default the optimizer now refuses to overwrite an existing bundle with the same basename. Re-run with `--overwrite` only when you intentionally want to replace prior artifacts.

The exported bundle contains:

- `<base>_final_params.csv`
- `<base>_parameters.csv`
- `<base>_losses.csv`
- `<base>_success_rates.csv`
- `<base>_metadata.json`
- `<base>_final_report.txt`

## Run the Batch

`run_batch.py` launches the four fixed cases used in the development workflow. For constant-velocity runs it now keeps the same seed for the paired `t=0.2` and `t=30.0` cases inside each domain, so time comparisons do not silently mix physics with different random initializations. Like the single-run CLI, it refuses to overwrite an existing bundle unless you pass `--overwrite`:

- `(domain=1, v=0.1, t=0.2)`
- `(domain=1, v=0.1, t=30.0)`
- `(domain=2, v=0.1, t=0.2)`
- `(domain=2, v=0.1, t=30.0)`

Run it with:

```bash
python run_batch.py --overwrite
```

The script writes results into `runs/` and copies `einstein_optimizer.py` there before execution.

## Generate Plots

To rebuild the paper-style figures for an existing run bundle:

```bash
python postprocess_plots.py --base domain_2_v_0p1_t_30p0
```

Optional arguments include:

- `--planes XY XZ`
- `--outdir <directory>`
- `--with-colorbar` / `--no-colorbar`
- `--dpi <int>`
- `--map-size <width> <height>`
- `--line-size <width> <height>`
- `--interface-buffer <float>`
- `--diagnostic-nxyz <int>`
- `--xlim <xmin> <xmax>`
- `--ylim <ymin> <ymax>`
- `--interpolation nearest|bilinear|bicubic`

`--diagnostic-nxyz` is especially useful when the optimizer had to run on a coarse grid for memory reasons. It keeps the optimized parameters fixed and recomputes the XY/XZ diagnostics on finer direct plane samplings, which is much closer to the old Mathematica workflow than slicing a coarse 3D cube. This makes the exported PNG figures much smoother without claiming a different optimization result.

`--xlim` and `--ylim` let you zoom to the physically relevant shell region, which is often far more informative than plotting the full `[-6, 6]` box. `--interpolation bilinear` can make display smoother in notebooks, but it only changes rendering, not the underlying diagnostic values.

The plotting pipeline reconstructs the field maps from the final optimized parameters using the same principal-stress eigenvalue diagnostics as the optimizer and the exported success-rate tables, via the shared helper module `physics_core.py`.

## Verify Outputs

To validate a completed run bundle and the expected plot set:

```bash
python verify_outputs.py --base domain_2_v_0p1_t_30p0
```

To compare the final parameters against the manuscript targets for the published `t=30.0`, `v=0.1` cases:

```bash
python compare_to_manuscript.py --base domain_2_v_0p1_t_30p0 --atol 0.10 --rtol 0.10
```

This prints the final `A`, `B`, and `R0` against the manuscript targets when those targets are defined. Use `--strict` if you want a nonzero exit code when the bundle lands outside tolerance.

The verifier checks:

- required run artifacts exist
- CSV headers match the revised pipeline
- success-rate columns use `nec,wec,dec,sec`
- final parameters match across CSV and JSON exports within tolerance
- the expected PNG figure set exists
- timestamps are sensible

The command returns a nonzero exit code on failure.

## Reproducibility Release

For the next journal submission, treat the public code-and-data archive as part of the paper package.
The repository now includes:

- `golden_dataset/`: regenerated optimized bundles for the core single-shell and double-shell cases
- `manuscript_target_bundles/`: direct evaluations of the manuscript target parameters under the same diagnostics
- `release_manifest.py`: helper that writes SHA256 manifests for the intended public release payload
- `REPRODUCIBILITY_RELEASE.md`: practical guidance for GitHub + Zenodo packaging
- `.zenodo.json`: editable metadata template for the Zenodo deposition

To generate a referee-facing manifest for the current release payload:

```bash
python release_manifest.py
```

This writes both JSON and Markdown manifests under `release_artifacts/`. Those files are useful when
preparing a Zenodo upload or a journal companion archive, because they make the payload explicit and
checkable.

## Legacy Audit Material

If you place historical notebook CSVs under `data/`, treat them as legacy audit artifacts rather than as referee-facing evidence. The legacy workflow typically exported:

- `<base>_parameters.csv`
- `<base>_losses.csv`
- `<base>_success_rates.csv`
- `<base>_violations.csv`

Those files usually use component-style success labels such as `h1..h4` instead of the revised `nec,wec,dec,sec` principal-stress diagnostics. The helper below compares legacy and/or revised runs numerically:

```bash
python compare_run_summaries.py --base-a data/domain2_t30.0 --base-b domain_2_v_0p1_t_30p0
```

Direct XY/XZ field summaries are produced only when both inputs are revised bundles with `metadata.json`.
## One-Shot End-to-End Helper

To run optimization, plotting, and verification in one command:

```bash
python generate_run_bundle.py --domain 2 --v 0.1 --t 30.0 --overwrite
```

For faster smoke tests in a limited environment, reduce the optimizer settings explicitly, for example:

```bash
python generate_run_bundle.py --domain 2 --v 0.1 --t 30.0 --overwrite --epochs 10 --n-xyz 12 --pretrain-trials 0
```

## Phase B Golden Dataset

The repository now includes a dedicated Phase B manifest for the small
referee-facing dataset that should be regenerated first:

- `core`: the single-shell and double-shell `t=30.0` reference bundles
- `time_audit`: optional `t=0.2` paired runs with the same seeds as the `t=30.0`
  cases, used to audit constant-velocity time-label handling
- `all`: both groups

Generate the core dataset with:

```bash
python run_golden_dataset.py --group core --outdir golden_dataset --overwrite
```

The case definitions live in `golden_dataset_cases.py`, and the workflow note is
documented in `docs/phase_b_golden_dataset.md`.

## Paper Figure Mapping

The post-processing filenames map directly onto the manuscript figure groups.

### Single-shell configuration (`domain_1_v_0p1_t_30p0`)

- Figure 1: `domain_1_v_0p1_t_30p0_a..._b..._R0..._XY_rho.png`, `..._XY_WECmin.png`, `..._XY_NECmin.png`
- Figure 2: `domain_1_v_0p1_t_30p0_a..._b..._R0..._XY_DECmargin.png`, `..._XY_SEC.png`
- Figure 3: `domain_1_v_0p1_t_30p0_a..._b..._R0..._XZ_rho.png`, `..._XZ_WECmin.png`, `..._XZ_NECmin.png`
- Figure 4: `domain_1_v_0p1_t_30p0_a..._b..._R0..._XZ_DECmargin.png`, `..._XZ_SEC.png`
- Figure 9 panels: `*_loss_components.png`, `*_success_fractions.png`, `*_params.png`

### Double-shell configuration (`domain_2_v_0p1_t_30p0`)

- Figure 5: `domain_2_v_0p1_t_30p0_a..._b..._R0..._XY_rho.png`, `..._XY_WECmin.png`, `..._XY_NECmin.png`
- Figure 6: `domain_2_v_0p1_t_30p0_a..._b..._R0..._XY_DECmargin.png`, `..._XY_SEC.png`
- Figure 7: `domain_2_v_0p1_t_30p0_a..._b..._R0..._XZ_rho.png`, `..._XZ_WECmin.png`, `..._XZ_NECmin.png`
- Figure 8: `domain_2_v_0p1_t_30p0_a..._b..._R0..._XZ_DECmargin.png`, `..._XZ_SEC.png`
- Figure 10 panels: `*_loss_components.png`, `*_success_fractions.png`, `*_params.png`

Field-map filenames follow the manuscript naming convention from the LaTeX source:

- `<base>_a{A}_b{B}_R0{R0}_XY_rho.png`
- `<base>_a{A}_b{B}_R0{R0}_XY_WECmin.png`
- `<base>_a{A}_b{B}_R0{R0}_XY_NECmin.png`
- `<base>_a{A}_b{B}_R0{R0}_XY_DECmargin.png`
- `<base>_a{A}_b{B}_R0{R0}_XY_SEC.png`
- `<base>_a{A}_b{B}_R0{R0}_XZ_rho.png`
- `<base>_a{A}_b{B}_R0{R0}_XZ_WECmin.png`
- `<base>_a{A}_b{B}_R0{R0}_XZ_NECmin.png`
- `<base>_a{A}_b{B}_R0{R0}_XZ_DECmargin.png`
- `<base>_a{A}_b{B}_R0{R0}_XZ_SEC.png`

For domain 1, the export still includes `R0...` with `R0=0.000` so the filenames remain consistent with the paper's LaTeX includes.

All field-map filenames use the revised manuscript language:

- `rho`
- `WECmin` (filename) / `WEC_min` (diagnostic label)
- `NECmin` (filename) / `NEC_min` (diagnostic label)
- `DECmargin` (filename) / `DEC_margin` (diagnostic label)
- `SEC`

### Display Styles for Field Maps

`postprocess_plots.py` provides two rendering modes for the same rebuilt fields:

- `--display-style default`: audit-facing output with minimal display processing. Use this when checking the raw sign structure or comparing directly against exported diagnostics.
- `--display-style old-paper`: manuscript-facing output with display-only smoothing, a zero-centered deadband, and isolated-pixel suppression to mimic the cleaner Mathematica `DensityPlot` appearance from the earlier draft.

This distinction is intentional. All quantitative claims in the manuscript should be read from the unsmoothed diagnostics (`success_rates.csv`, `losses.csv`, `final_params.csv`, and raw rebuilt fields). The `old-paper` preset exists only to improve figure readability and does not feed back into the optimization or into any reported percentages/margins.

### Mathematica Final Rendering

If the manuscript-facing Python PNGs are still not visually satisfactory, the
recommended fallback is to keep Python as the audited source of truth and use
Mathematica only as the final renderer.

The repository now includes:

- `export_mathematica_plane_maps.py`: rebuilds XY/XZ diagnostic planes from the
  cleaned bundle and exports them as Mathematica-friendly JSON grids.
- `render_mathematica_plane_maps.py`: Python port of the Mathematica
  presentation renderer. It can render either an existing
  `mathematica_exports/<bundle>/` JSON directory or rebuild directly from a run
  bundle with manuscript-oriented defaults (`plane_n=700`, `bicubic`
  interpolation, robust zero-centered clipping, and the same five-stop
  diverging palette).
- `render_mathematica_plane_maps.wl`: a Wolfram Language renderer that imports
  those JSON files and creates clean `ListDensityPlot` manuscript panels with a
  five-stop diverging palette similar to the original draft notebooks.

If you want a pure-Python manuscript-style render directly from a run bundle,
start with:

```bash
python render_mathematica_plane_maps.py \
  --base manuscript_target_bundles/domain_2_v_0p1_t_30p0 \
  --output-dir python_rendered \
  --planes XY XZ
```

This is usually the best first option when the standard `postprocess_plots.py`
 output looks too coarse or too pixelated. The rendering is still
 presentation-only; it does not modify the underlying diagnostic values.

Typical workflow:

```bash
python export_mathematica_plane_maps.py \
  --base manuscript_target_bundles/domain_2_v_0p1_t_30p0 \
  --outdir mathematica_exports \
  --planes XY XZ \
  --plane-n 700
```

This writes a bundle-specific export directory containing:

- `manifest.json`
- one JSON file per plane/field combination
- conservative display-limit hints for Mathematica

You can then render those files either from the Mathematica front end or with
WolframScript:

```bash
wolframscript -file render_mathematica_plane_maps.wl -- \
  --input-dir mathematica_exports/domain_2_v_0p1_t_30p0_a1p848_b1p135_R01p292 \
  --output-dir mathematica_exports/domain_2_v_0p1_t_30p0_a1p848_b1p135_R01p292/rendered \
  --image-size 420 \
  --plot-min -2.6 \
  --plot-max 2.6
```

If you already have the JSON export directory and only want the Python port of
the presentation layer, use:

```bash
python render_mathematica_plane_maps.py \
  --input-dir mathematica_exports/domain_2_v_0p1_t_30p0_a1p848_b1p135_R01p292
```

Important: this Mathematica path is presentation-only. The paper's success
fractions, margins, and optimization diagnostics must still be quoted from the
Python-exported raw bundle data.

## Reproducibility Notes

### Origin of the Code

The project originated in an interactive Google Cloud notebook workflow used during the paper development stage. The repository version consolidates that workflow into standalone Python scripts so that optimization, post-processing, and verification can be rerun without the original notebook environment.

A short Phase A audit note for referee preparation is included at `docs/phase_a_referee_audit.md`. That document explains how the revised pipeline differs from the historical notebook exports and why the legacy CSVs in `data/` should not be read as direct support for the revised principal-stress claims.

### Python and TensorFlow Environment

The scripts were tested in the following environment:

- Python `3.9.19`
- NumPy `1.26.4`
- Matplotlib `3.9.2`
- TensorFlow `2.19.0`
- tqdm `4.66.5`
- Platform: Windows 10 (`10.0.26200`)

`einstein_optimizer.py` explicitly sets:

- `TF_CPP_MIN_LOG_LEVEL=2`
- `TF_ENABLE_ONEDNN_OPTS=0`
- `TF_DETERMINISTIC_OPS=1`
- TensorFlow default float type to `float32`

### Exported CSV and Metadata Scheme

For each run base `<base>`, the optimizer exports:

- `<base>_final_params.csv`
  Header: `A,B,R0,alpha`
- `<base>_parameters.csv`
  Header: `epoch,A,B,alpha` for domain 1 or `epoch,A,B,R0,alpha` for domain 2
- `<base>_losses.csv`
  Header: `epoch,total_loss,physics_loss,reg_breach,inv_alpha,E_soft_%,shear_pen`
- `<base>_success_rates.csv`
  Header: `epoch,nec,wec,dec,sec,nec_margin_soft,wec_margin_soft,dec_margin_soft,sec_margin_soft`
- `<base>_metadata.json`
  Includes `domain_type`, `v_mode`, `velocity`, `time`, `N_xyz`, `xyz_range`, and `final_parameters`

The revised pipeline uses principal-stress eigenvalue diagnostics in the orthonormal bubble-centered/comoving frame. Success columns are `nec`, `wec`, `dec`, and `sec`; legacy `h1`-`h6` naming is not used.

Only the seed's non-negative `rho` is treated as an analytic construction feature. Global WEC/NEC/DEC/SEC satisfaction is not enforced by identity; those margins are evaluated numerically from the principal stresses and penalized in the optimization objective.

### Plotting Pipeline

`postprocess_plots.py` reloads the exported metadata and final parameters, reconstructs the final field snapshot through `EinsteinTrainerCPU`, and then evaluates:

- `rho`
- `WECmin` / `WEC_min`
- `NECmin` / `NEC_min`
- `DECmargin` / `DEC_margin`
- `SEC`

The same principal-stress definitions are used for both field maps and success plots. Invalid values are masked with `NaN` rather than silently zeroed.

For constant-velocity runs with fixed topology, the paper interprets the diagnostics in bubble-centered/comoving coordinates. Under those assumptions, changing the time label changes only the lab-frame displacement and does not create a physically distinct comoving field map.

## Suggested Zenodo Release Checklist

Before creating a Zenodo archive, review the following:

1. Fill the placeholders in `CITATION.cff` and confirm the citation metadata.
2. Confirm the license choice in `LICENSE` is the one you want to publish under.
3. Remove or regenerate any local smoke-test outputs that should not be part of the source release.
4. Tag the repository state used for the paper release.
5. Create a GitHub release (or equivalent VCS release) from that tag.
6. Enable Zenodo archiving for the repository.
7. Add the Zenodo DOI back into `CITATION.cff` and `README.md` after Zenodo assigns it.
8. Record the exact environment used for the archival release if it differs from the tested environment listed above.

## Placeholders to Review Before Release

The following files contain placeholders that should be updated before a final public archival release:

- `CITATION.cff`: author, repository URL, DOI, and release metadata





