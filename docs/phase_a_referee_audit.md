# Phase A Audit: Referee-Facing Pipeline

This repository now distinguishes between two classes of artifacts:

- revised pipeline: the code path intended for manuscript reproduction and referee sharing
- legacy notebook artifacts: historical CSVs and plots kept only for audit and comparison

## What the revised pipeline is

The active scientific workflow is:

- `einstein_optimizer.py`
- `postprocess_plots.py`
- `verify_outputs.py`
- `generate_run_bundle.py`

This workflow is **not a PINN**. It is a parametric optimization over an analytic seed ansatz, with physics-informed penalty terms and principal-stress diagnostics evaluated in the bubble-centered/comoving frame.

## What is guaranteed analytically, and what is not

The intended analytic statement is limited:

- the seed is constructed so that `rho >= 0`
- under uniform constant translation, the optimization seeks configurations with small residual deficits in the energy-condition margins

The revised code does **not** guarantee global WEC/NEC/DEC/SEC satisfaction by construction. Those conditions are evaluated numerically from principal-stress eigenvalues and penalized in the loss.

## Legacy workflow status

Historical CSVs under `data/` come from an earlier notebook workflow and use a different export scheme. Typical legacy files are:

- `<base>_parameters.csv`
- `<base>_losses.csv`
- `<base>_success_rates.csv`
- `<base>_violations.csv`

Legacy success tables use component-style labels such as `h1..h4`, not the revised `nec,wec,dec,sec` scheme. They should therefore be treated as audit material only, not as direct support for principal-stress claims in the revised manuscript.

## Important batch-seed caveat

The old batch workflow varied the random seed with the time label. For constant-velocity runs, that means changing `t` also changed the initialization, so old `t=0.2 / 1.0 / 10.0 / 30.0` sweeps are not clean evidence of physical time dependence.

The cleaned `run_batch.py` in this repository now keeps the same seed for the paired time runs within each domain to avoid that confound.

## Comparison helper

Use `compare_run_summaries.py` to compare:

- revised vs revised bundles
- legacy vs legacy bundles
- legacy vs revised bundles

Examples:

```bash
python compare_run_summaries.py --base-a data/domain2_t30.0 --base-b domain_2_v_0p1_t_30p0
python compare_run_summaries.py --base-a data/domain1_t30.0 --base-b data/domain1_t10.0
```

Direct XY/XZ field summaries are available only when both inputs are revised bundles with `metadata.json`.

## Referee-facing recommendation

For manuscript figures and quantitative claims, use only revised bundles that provide:

- `final_params.csv`
- `parameters.csv`
- `losses.csv`
- `success_rates.csv`
- `metadata.json`

Use the legacy data only to document how the workflow evolved or to explain why older notebook plots are not authoritative for the revised text.
