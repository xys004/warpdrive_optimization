# Phase C Audit: `domain_1_v_0p1_t_30p0`

This note records what the regenerated `domain_1` golden-dataset bundle supports, what it does not support, and which manuscript statements should be revised before sharing the package with referees.

## Scope

- Regenerated bundle:
  - `golden_dataset/domain_1_v_0p1_t_30p0_*`
- Manuscript under review:
  - `C:\Users\Nelson\Downloads\warpdrive_clean_submission.tex`
- This is a **single-case audit** for the single-shell topology at `v = 0.1`, `t = 30.0`.

## Regenerated bundle summary

Final parameters from `domain_1_v_0p1_t_30p0_final_params.csv`:

- `A = 1.050044298171997`
- `B = 1.6419425010681152`
- `alpha = 222.69439697265625`

Final success fractions from `domain_1_v_0p1_t_30p0_success_rates.csv`:

- `NEC = 32.70%`
- `WEC = 30.23%`
- `DEC = 1.35%`
- `SEC = 66.17%`

Final soft margins:

- `nec_margin_soft = -2.063e-01`
- `wec_margin_soft = -2.063e-01`
- `dec_margin_soft = -4.206e-01`
- `sec_margin_soft = 1.643e-02`

Final losses from `domain_1_v_0p1_t_30p0_losses.csv`:

- `total_loss = 4.4905`
- `physics_loss = 7.32e-13`
- `reg_breach = 0`
- `inv_alpha = 4.4905`

## Plane-map ranges

### XY plane

- `rho`: min `-4.424e-06`, max `1.413e-01`
- `WEC_min`: min `-4.305e+00`, max `1.937e-07`
- `NEC_min`: min `-4.305e+00`, max `1.937e-07`
- `DEC_margin`: min `-4.305e+00`, max `-3.137e-06`
- `SEC`: min `-8.853e+00`, max `8.337e+00`

### XZ plane

- `rho`: min `-4.424e-06`, max `1.413e-01`
- `WEC_min`: min `-1.287e+01`, max `4.038e-02`
- `NEC_min`: min `-1.287e+01`, max `4.038e-02`
- `DEC_margin`: min `-1.287e+01`, max `1.821e-03`
- `SEC`: min `-2.590e+01`, max `2.218e+01`

These ranges show that the regenerated single-shell case is **not** in the near-saturated WEC/NEC regime described by the current manuscript, and that the SEC trace is sign-changing on both sampled slices.

## Supported claims

The following statements remain supported by the regenerated `domain_1` bundle:

- `DEC` is again the tightest residual constraint.
- `SEC` is easier to satisfy than `DEC` in aggregate.
- The optimizer produces a stable parameter snapshot and a valid verified bundle.

## Unsupported or overstated claims

The following manuscript claims are too strong for this regenerated `domain_1` case:

- "the slice is everywhere non-negative to numerical accuracy" for `WEC_min`
  - The regenerated success fractions and map ranges do not support this.
- "same shell-localized near-saturation pattern as the WEC/NEC"
  - The regenerated WEC/NEC maps are visibly mixed and not close to full saturation.
- "SEC stays positive throughout the shell"
  - The regenerated `SEC` plots are sign-changing.
- "the optimizer primarily reduces the physics loss"
  - As in `domain_2`, the visible late-time loss budget is dominated by `inv_alpha`.
- "both plateau after an initial transient"
  - The parameter trajectories are almost flat for the entire exported run, which is more naturally explained by the pretraining stage than by a large main-run transient.

## Manuscript lines that should be revised

The file under review is:

- `C:\Users\Nelson\Downloads\warpdrive_clean_submission.tex`

### Global narrative

Statements that should be softened or made conditional:

- Line 318: the current training-curve summary overstates high `NEC/WEC` success fractions for the single-shell case as well.
- Line 370: the current paragraph is much stronger than the regenerated bundle supports.
- Line 383: the general conclusion about low-velocity optimized configurations is currently too optimistic for the regenerated single-shell case.

### Figure captions: single-shell slices

- Line 403 (`d1 XY WEC_min`)
  - The regenerated run does not support "everywhere non-negative to numerical accuracy".
- Line 412 (`d1 XY NEC_min`)
  - The regenerated run does not support "same shell-localized near-saturation pattern".
- Line 435 (`d1 XY SEC`)
  - The regenerated `SEC` is not positive throughout the shell.
- Line 457 (`d1 XZ WEC_min`)
  - The current caption understates the magnitude of the regenerated deficits.
- Line 466 (`d1 XZ NEC_min`)
  - Same issue as `WEC_min`.
- Line 489 (`d1 XZ SEC`)
  - The regenerated meridional SEC slice is sign-changing, so the current wording should be replaced.

### Figure captions: single-shell optimization

- Line 626 (`d1 success fractions`)
  - The relative ordering `DEC < WEC/NEC < SEC` is supported.
  - The current caption should avoid implying a high-success regime for `NEC/WEC`.
- Line 635 (`d1 params`)
  - The exported run does not show a strong transient; it shows near-stationarity.
- Line 640 (`d1 optimizer summary`)
  - The optimizer summary should be reframed around stabilization rather than dramatic improvement of the physics term.

## Suggested replacement language

These are proposed rewrites for the `domain_1` parts only.

### Single-shell discussion around line 370

Suggested replacement:

> For the single-shell configuration, the regenerated bundle still shows `\DEC` as the dominant residual obstruction, but the `\WEC/\NEC` margins are not in a near-saturated regime across the sampled slices. The `\SEC` trace is mixed rather than uniformly positive. The main qualitative lesson is therefore not that the single shell fully preserves the weak or null conditions, but that its residual deficits remain simpler and easier to organize than in the double-shell topology.

### Caption for line 403 (`d1 XY WEC_min`)

Suggested replacement:

> Minimum weak-energy-condition margin `\mathrm{WEC}_{\min}`; the regenerated slice contains localized negative sectors rather than remaining everywhere non-negative, but still stays less restrictive than the corresponding `\DEC` diagnostic.

### Caption for line 412 (`d1 XY NEC_min`)

Suggested replacement:

> Minimum null-energy-condition margin `\mathrm{NEC}_{\min}`, showing a similar localized deficit structure to the WEC diagnostic.

### Caption for line 435 (`d1 XY SEC`)

Suggested replacement:

> Strong-energy-condition trace on the equatorial slice; the regenerated pattern is sign-changing, although it remains less restrictive in aggregate than the `\DEC` margin.

### Caption for line 489 (`d1 XZ SEC`)

Suggested replacement:

> `\SEC` on the meridional slice; the trace is mixed rather than uniformly positive, but the aggregate SEC success fraction still exceeds the DEC success fraction.

### Caption for line 626 (`d1 success fractions`)

Suggested replacement:

> Fraction of grid points satisfying NEC/WEC/DEC/SEC; SEC remains the easiest condition to satisfy, DEC remains the tightest global constraint, and NEC/WEC settle at intermediate but still significantly constrained levels.

### Caption for line 635 (`d1 params`)

Suggested replacement:

> Convergence of seed parameters `A` and `B` for the single-shell seed; the main run remains almost stationary, consistent with a pretraining stage that already selected a near-plateau configuration before the published optimization history begins.

## Editorial recommendation

For `domain_1`, the regenerated evidence suggests a stronger revision than for `domain_2`:

- keep the claim that `DEC` is the dominant residual obstruction,
- keep the claim that the single-shell topology is simpler than the double-shell one,
- **remove** claims that `WEC/NEC` are nearly saturated on the sampled slices,
- **remove** claims that `SEC` is positive throughout the shell,
- reframe the training discussion around a preconditioned, almost-stationary main run rather than a visibly improving transient.

## Next task

Use this audit together with `phase_c_domain2_audit.md` to prepare a cross-topology manuscript revision plan. The current paper language should not be updated piecemeal; it should be revised once both regenerated topologies have been considered together.
