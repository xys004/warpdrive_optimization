# Reproducibility and Zenodo Release Plan

This note describes how to prepare a referee-facing public archive for the revised warp-drive study.
The goal is that an external reviewer can download one release, inspect the code and data, and
reproduce the manuscript-facing bundles without needing hidden notebook history.

## Release Principle

The public package should contain:

1. the cleaned source code used for optimization, diagnostics, plotting, and verification;
2. the regenerated golden bundles used to support the manuscript narrative;
3. the direct manuscript-target evaluations used to audit the published parameter values;
4. a manifest with hashes so the release payload is explicit and checkable;
5. brief documentation stating what is authoritative and what is legacy.

Historical notebook exports under `data/` should not be the primary referee-facing evidence unless
there is a specific reason to publish them as legacy audit material.

## Recommended Public Payload

### Source code

Keep the GitHub release source archive focused on the active workflow:

- `physics_core.py`
- `einstein_optimizer.py`
- `postprocess_plots.py`
- `verify_outputs.py`
- `generate_run_bundle.py`
- `run_golden_dataset.py`
- `evaluate_manuscript_targets.py`
- `compare_run_summaries.py`
- `README.md`
- `requirements.txt`
- `LICENSE`
- `CITATION.cff`

### Data bundles

The minimal referee-facing data package should include:

- `golden_dataset/`
  - optimized `domain_1_v_0p1_t_30p0`
  - optimized `domain_2_v_0p1_t_30p0`
- `manuscript_target_bundles/`
  - direct evaluation of the manuscript target parameters for the same two cases

Each bundle should include:

- `*_final_params.csv`
- `*_parameters.csv`
- `*_losses.csv`
- `*_success_rates.csv`
- `*_metadata.json`
- `*_final_report.txt`
- the PNG diagnostics generated from the same bundle

### Manuscript audit companions

Include the compact manuscript-side notes if you want reviewers to see how the wording was aligned
with the regenerated evidence:

- `phase_c_domain1_audit.md`
- `phase_c_domain2_audit.md`
- `phase_c_synthesis.md`
- `manuscript_revision_explained.tex`
- `manuscript_revision_map.tex`
- `referee_revision_summary.tex`

## Recommended Archive Layout

A practical layout for Zenodo is:

```text
warp_optimization_release/
|-- source/                     # GitHub release snapshot or source zip
|-- golden_dataset/             # optimized referee-facing bundles
|-- manuscript_target_bundles/  # fixed-parameter manuscript checks
|-- release_artifacts/
|   |-- release_manifest.json
|   `-- release_manifest.md
|-- README.md
`-- CITATION.cff
```

You can implement this either as:

- a single Zenodo deposition containing both source and data, or
- a GitHub release mirrored to Zenodo plus one additional Zenodo upload containing the generated
  bundles and manifest.

For a journal submission, either is fine as long as the manuscript points to one stable DOI and the
archive content is clearly described.

## Release Workflow

1. Confirm the repository is on the intended commit.
2. Regenerate the authoritative bundles if needed.
3. Run `verify_outputs.py` on each authoritative bundle.
4. Generate a manifest with `release_manifest.py`.
5. Create a GitHub release tag.
6. Archive the release in Zenodo.
7. Update `CITATION.cff` with the final DOI.
8. Update the manuscript data/code availability statement with the DOI and a short description of
   what the archive contains.

## Manifest Generation

Generate a hash manifest for the referee-facing release package with:

```bash
python release_manifest.py
```

This writes:

- `release_artifacts/release_manifest.json`
- `release_artifacts/release_manifest.md`

The default manifest includes the cleaned scripts, the optimized golden bundles, the manuscript
parameter bundles, and the manuscript-audit companions.

If you want a narrower manifest, specify the paths explicitly, for example:

```bash
python release_manifest.py --path golden_dataset --path manuscript_target_bundles --path README.md
```

## Zenodo Metadata

A template `.zenodo.json` is provided in the repository. Before creating the final release, update:

- title
- description
- creators
- affiliations
- keywords
- related identifiers if you want to point back to the manuscript preprint or journal page

## What the Manuscript Should Promise

For the next submission, the manuscript should promise only what the archive actually provides:

- the cleaned optimization and plotting code,
- the exact bundles used for the main topology-level claims,
- direct evaluations of the manuscript target parameters,
- documentation describing the scope and limits of the workflow.

This is stronger and safer than promising raw notebook history or every experimental intermediate.

## Suggested Availability Statement

A suitable data/code availability statement can be adapted from the following:

> The cleaned optimization, plotting, and verification code used in this study is archived in a
> public repository with DOI-backed release metadata. The archive also includes the regenerated
> single-shell and double-shell bundles used for the manuscript-level claims, together with direct
> evaluations of the manuscript target parameters under the same diagnostics. These materials are
> sufficient to reproduce the reported bundle summaries and field-map figures.

## Final Recommendation

For the next journal submission, treat the public archive as part of the scientific argument. The
referee should be able to:

1. download the archive,
2. inspect the exact bundles,
3. verify the outputs,
4. regenerate plots from the stored parameters,
5. and understand which results are optimized runs versus fixed manuscript-target evaluations.

That level of clarity will do much more for the next review round than trying to defend the older,
less auditable notebook-era workflow.
