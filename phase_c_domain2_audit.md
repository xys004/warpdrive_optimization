# Phase C Audit: `domain_2_v_0p1_t_30p0`

This note records what the regenerated `domain_2` golden-dataset bundle supports, what it does not support, and which manuscript statements should be revised before sharing the package with referees.

## Scope

- Regenerated bundle:
  - `golden_dataset/domain_2_v_0p1_t_30p0_*`
- Manuscript under review:
  - `C:\Users\Nelson\Downloads\warpdrive_clean_submission.tex`
- This is a **single-case audit** for the double-shell topology at `v = 0.1`, `t = 30.0`.
- Single-shell claims still need a corresponding Phase C pass once the `domain_1` golden bundle is regenerated.

## Regenerated bundle summary

Final parameters from `domain_2_v_0p1_t_30p0_final_params.csv`:

- `A = 1.0426318645477295`
- `B = 1.314130187034607`
- `R0 = 1.264353632926941`
- `alpha = 223.19937133789062`

Final success fractions from `domain_2_v_0p1_t_30p0_success_rates.csv`:

- `NEC = 40.13%`
- `WEC = 40.13%`
- `DEC = 9.87%`
- `SEC = 74.34%`

Final soft margins:

- `nec_margin_soft = -6.396e-04`
- `wec_margin_soft = -6.396e-04`
- `dec_margin_soft = -4.386e-03`
- `sec_margin_soft = 7.681e-03`

Final losses from `domain_2_v_0p1_t_30p0_losses.csv`:

- `total_loss = 4.4803`
- `physics_loss = 4.97e-15`
- `reg_breach = 0`
- `inv_alpha = 4.4803`

## Plane-map ranges

The rerendered plane maps were sampled directly from the regenerated bundle.

### XY plane

- `rho`: min `3.276e-01`, max `6.505e-01`
- `WEC_min`: min `-3.746e+00`, max `-5.603e-06`
- `NEC_min`: min `-3.746e+00`, max `-5.603e-06`
- `DEC_margin`: min `-3.746e+00`, max `-2.235e-04`
- `SEC`: min `-8.228e+00`, max `8.145e+00`

### XZ plane

- `rho`: min `3.276e-01`, max `6.505e-01`
- `WEC_min`: min `-9.892e+01`, max `8.834e-02`
- `NEC_min`: min `-9.892e+01`, max `8.834e-02`
- `DEC_margin`: min `-9.892e+01`, max `8.834e-02`
- `SEC`: min `-1.987e+02`, max `1.915e+02`

These ranges matter because they show that the regenerated double-shell `SEC` is **not** non-negative everywhere on the sampled slices, and that `WEC/NEC` are not in a "high success fraction" regime for this run.

## Supported claims

The following statements are supported by the regenerated `domain_2` bundle:

- `rho` remains positive on the sampled slices.
- `DEC` is the tightest residual constraint among the principal-stress diagnostics.
- `SEC` is easier to satisfy than `DEC` in aggregate, as shown by the final success fractions (`74.34%` versus `9.87%`).
- `WEC/NEC` and `DEC` residual deficits remain localized rather than filling the full box.
- The double-shell topology is more constrained than the single-shell narrative suggests in the manuscript.

## Unsupported or overstated claims

The following manuscript claims are too strong for this regenerated `domain_2` case:

- "NEC and WEC reach high success fractions"
  - The regenerated bundle ends at `40.13%`, which is not a high-success regime.
- "SEC remains positive throughout" or equivalent wording
  - The regenerated `SEC` plots contain both positive and negative regions.
  - The sampled plane-map ranges confirm sign changes in both `XY` and `XZ`.
- "The optimizer primarily reduces the physics loss"
  - In this run the final loss budget is dominated by `inv_alpha`, while `physics_loss` is numerically tiny throughout the exported history.
  - The loss plot does not visually support the current prose emphasis.
- "Parameters plateau after an initial transient"
  - The parameter plot shows only a modest early adjustment and then a long stable regime.
  - The current wording overstates the presence of a pronounced transient.

## Manuscript lines that should be revised

The file under review is:

- `C:\Users\Nelson\Downloads\warpdrive_clean_submission.tex`

### Global narrative

Statements that should be softened or made conditional:

- Line 318: the current training-curve summary overstates high `NEC/WEC` success fractions and the role of `physics_loss`.
- Line 372: "positive SEC" is too strong for the regenerated double-shell case.
- Line 383: "typically exhibit a non-negative SEC trace on the plotted slices" is not supported by this regenerated `domain_2` bundle.

### Figure captions: double-shell slices

- Line 544 (`d2 XY SEC`)
  - Current wording says the condition remains positive throughout.
  - This is contradicted by the regenerated `XY_SEC` plot and the plane-map range.
- Line 599 (`d2 XZ SEC`)
  - Current wording says the positive condition indicates the strong energy condition is globally easier to satisfy.
  - The aggregate part ("easier than DEC") is acceptable, but the local positivity wording is too strong.

### Figure captions: double-shell optimization

- Line 662 (`d2 success fractions`)
  - "NEC and WEC reach high success fractions" should be replaced.
  - The data support "NEC and WEC remain above DEC but below full saturation."
- Line 671 (`d2 params`)
  - The plateau claim is broadly acceptable, but the language should avoid implying a large transient.
- Line 676 (`d2 optimizer summary`)
  - The "more strongly constrained solution" framing is good and should be kept.
  - It should explicitly mention that the surviving deficits are still significant for `WEC/NEC`.

## Suggested replacement language

These are proposed rewrites for the `domain_2` parts only.

### Main text around line 318

Suggested replacement:

> Representative training curves are shown in Figs.~\\ref{fig:d1-opt} and~\\ref{fig:d2-opt}. The total loss decreases and then stabilizes as the interface regularity term is suppressed and the inverse-width regularizer controls the late-time balance. The NEC/WEC/DEC/SEC success fractions settle into topology-dependent plateaus, with the double-shell case remaining noticeably more constrained than the single-shell case. The parameter trajectories show that the main run stays in a narrow region of parameter space after a modest early adjustment, which we take as a practical convergence signal for the chosen diagnostics and loss weights.

### Double-shell discussion around line 372

Suggested replacement:

> The double-shell configuration presents a more complex geometry. It maintains `\\rho > 0` on the sampled slices, but localized `\\WEC/\\NEC` deficits emerge near the geometric center or the inner vacuum interface. The regenerated `\\SEC` diagnostics are mixed rather than uniformly positive, although they remain less restrictive in aggregate than the `\\DEC`. As in the single-shell case, the `\\DEC` remains the tightest constraint.

### Caption for line 544 (`d2 XY SEC`)

Suggested replacement:

> `\\SEC` on the equatorial slice; the trace is mixed rather than uniformly positive, but its deficits are milder in aggregate than the corresponding `\\DEC` deficits.

### Caption for line 599 (`d2 XZ SEC`)

Suggested replacement:

> `\\SEC` on the same slice; the pattern is sign-changing but remains less restrictive in aggregate than the `\\DEC`, consistent with the higher final SEC success fraction.

### Caption for line 662 (`d2 success fractions`)

Suggested replacement:

> Fraction of grid points satisfying NEC/WEC/DEC/SEC; the double-shell case settles into a constrained regime in which SEC remains the easiest condition to satisfy, NEC/WEC stay above DEC, and DEC remains the tightest global constraint.

### Caption for line 671 (`d2 params`)

Suggested replacement:

> Convergence of seed parameters `A`, `B`, and `R_0`; after a modest early adjustment the main run remains in a narrow parameter band, indicating a stable double-shell configuration within the adopted loss and diagnostic choices.

## Editorial recommendation

For `domain_2`, the manuscript should be revised toward the following posture:

- keep the claim that `DEC` is the dominant residual obstruction,
- keep the claim that `rho` stays positive on the sampled slices,
- keep the claim that double shells are more constrained,
- **drop or soften** claims that `WEC/NEC` are in a high-success regime,
- **drop or soften** claims that `SEC` is positive throughout the sampled slices,
- reframe the optimizer discussion in terms of stabilization and trade-offs rather than dramatic early loss reduction.

## Next Phase C task

Repeat the same audit for the regenerated `domain_1` golden bundle before editing the manuscript globally. The final paper language should only make topology-specific claims that are directly supported by the corresponding regenerated bundle.
