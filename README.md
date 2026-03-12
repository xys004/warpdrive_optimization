# warp_optimization

`warp_optimization` is a TensorFlow-based constrained-optimization and post-processing workflow for the revised warp-drive energy-condition study. The repository packages the optimizer, plot generation, and output verification steps used to reproduce the paper's principal-stress diagnostics for uniformly translated, zero-vorticity single-shell and double-shell seed configurations.

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

The included notebook `colab_smoke_test.ipynb` is the easiest cloud workflow.

1. Open `colab_smoke_test.ipynb` from the GitHub repository in Google Colab.
2. Because the repository is private, paste a GitHub token into the `GITHUB_TOKEN` variable in the first code cell.
3. Run the notebook cells to clone the repo, install dependencies, generate a reduced smoke-test bundle, and verify the outputs.
4. If the smoke test passes, rerun `generate_run_bundle.py` from Colab with larger `--epochs`, `--n-xyz`, and pretraining settings.

If you prefer not to store a token in the notebook, you can also clone interactively in a Colab terminal/session or make the repository public later for easier notebook launching.

## Run a Single Optimization

To run one optimization snapshot and export the raw run bundle:

```bash
python einstein_optimizer.py --domain 2 --v 0.1 --t 30.0 --seed 47
```

This writes files with a base name of the form:

```text
domain_<domain>_v_<velocity tag>_t_<time tag>
```

For example:

```text
domain_2_v_0p1_t_30p0
```

The exported bundle contains:

- `<base>_final_params.csv`
- `<base>_parameters.csv`
- `<base>_losses.csv`
- `<base>_success_rates.csv`
- `<base>_metadata.json`
- `<base>_final_report.txt`

## Run the Batch

`run_batch.py` launches the four fixed cases used in the development notebook workflow:

- `(domain=1, v=0.1, t=0.2)`
- `(domain=1, v=0.1, t=30.0)`
- `(domain=2, v=0.1, t=0.2)`
- `(domain=2, v=0.1, t=30.0)`

Run it with:

```bash
python run_batch.py
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

The plotting pipeline reconstructs the field maps from the final optimized parameters using the same principal-stress eigenvalue diagnostics as the optimizer and the exported success-rate tables, via the shared helper module `physics_core.py`.

## Verify Outputs

To validate a completed run bundle and the expected plot set:

```bash
python verify_outputs.py --base domain_2_v_0p1_t_30p0
```

The verifier checks:

- required run artifacts exist
- CSV headers match the revised pipeline
- success-rate columns use `nec,wec,dec,sec`
- final parameters match across CSV and JSON exports within tolerance
- the expected PNG figure set exists
- timestamps are sensible

The command returns a nonzero exit code on failure.

## One-Shot End-to-End Helper

To run optimization, plotting, and verification in one command:

```bash
python generate_run_bundle.py --domain 2 --v 0.1 --t 30.0
```

For faster smoke tests in a limited environment, reduce the optimizer settings explicitly, for example:

```bash
python generate_run_bundle.py --domain 2 --v 0.1 --t 30.0 --epochs 10 --n-xyz 12 --pretrain-trials 0
```

## Paper Figure Mapping

The post-processing filenames map directly onto the manuscript figure groups.

### Single-shell configuration (`domain_1_v_0p1_t_30p0`)

- Figure 1: `*_XY_rho.png`, `*_XY_WECmin.png`, `*_XY_NECmin.png`
- Figure 2: `*_XY_DECmargin.png`, `*_XY_SEC.png`
- Figure 3: `*_XZ_rho.png`, `*_XZ_WECmin.png`, `*_XZ_NECmin.png`
- Figure 4: `*_XZ_DECmargin.png`, `*_XZ_SEC.png`
- Figure 9 panels: `*_loss_components.png`, `*_success_fractions.png`, `*_params.png`

### Double-shell configuration (`domain_2_v_0p1_t_30p0`)

- Figure 5: `*_XY_rho.png`, `*_XY_WECmin.png`, `*_XY_NECmin.png`
- Figure 6: `*_XY_DECmargin.png`, `*_XY_SEC.png`
- Figure 7: `*_XZ_rho.png`, `*_XZ_WECmin.png`, `*_XZ_NECmin.png`
- Figure 8: `*_XZ_DECmargin.png`, `*_XZ_SEC.png`
- Figure 10 panels: `*_loss_components.png`, `*_success_fractions.png`, `*_params.png`

All field-map filenames use the revised manuscript language:

- `rho`
- `WECmin` (filename) / `WEC_min` (diagnostic label)
- `NECmin` (filename) / `NEC_min` (diagnostic label)
- `DECmargin` (filename) / `DEC_margin` (diagnostic label)
- `SEC`

## Reproducibility Notes

### Origin of the Code

The project originated in an interactive Google Cloud notebook workflow used during the paper development stage. The repository version consolidates that workflow into standalone Python scripts so that optimization, post-processing, and verification can be rerun without the original notebook environment.

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

