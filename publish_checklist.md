# Publish Checklist

Use this checklist when you are ready to push `warp_optimization` to GitHub and prepare a public release.

## 1. Final sanity check

- Confirm `README.md` matches the current code and output filenames.
- Confirm `requirements.txt` matches the environment you want to support.
- Confirm `LICENSE` is the license you want to publish under.
- Fill in the placeholders in `CITATION.cff`.
- Decide whether any local generated outputs should be deleted before the first push.

## 2. Clean local status

Run:

```bash
git status --short
```

Expected source files for the initial upload include:

- `.gitignore`
- `README.md`
- `requirements.txt`
- `LICENSE`
- `CITATION.cff`
- `physics_core.py`
- `einstein_optimizer.py`
- `run_batch.py`
- `postprocess_plots.py`
- `verify_outputs.py`
- `generate_run_bundle.py`
- `colab_smoke_test.ipynb`
- `publish_checklist.md`
- `COMMIT_MESSAGE_SUGGESTIONS.md`

## 3. Create the repository on GitHub

Create a repository under your GitHub account.

Important: `https://github.com/xys004` is your GitHub account page, not a repository URL.
You will need a full repository URL of the form:

```text
https://github.com/xys004/<repo-name>.git
```

## 4. Add the remote and push

Example:

```bash
git remote add origin https://github.com/xys004/<repo-name>.git
git push -u origin master
```

If the remote already exists, check it with:

```bash
git remote -v
```

## 5. Verify GitHub contents

After the push:

- Open the repository page and confirm the source files are present.
- Open `README.md` in GitHub and make sure the formatting renders cleanly.
- Open `colab_smoke_test.ipynb` from GitHub and confirm Colab can load it.

## 6. Optional release tag

When you are happy with the public source snapshot:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Adjust the tag name to match your release plan.

## 6.5 Build a release manifest

Before creating the public archive, generate a manifest of the files you intend to publish:

```bash
python release_manifest.py
```

This writes:

- `release_artifacts/release_manifest.json`
- `release_artifacts/release_manifest.md`

Review those files before publishing. They are useful for Zenodo and for checking that you are
sharing the cleaned bundles (`golden_dataset/`, `manuscript_target_bundles/`) rather than the older
legacy notebook exports by mistake.

## 7. Zenodo preparation

- Connect the GitHub repository to Zenodo.
- Create a GitHub release from the intended tag.
- Let Zenodo archive the release.
- Copy the assigned DOI back into `CITATION.cff` and `README.md`.

## 8. Colab smoke test after publish

From GitHub, open `colab_smoke_test.ipynb` in Colab and run the reduced workflow:

```bash
python generate_run_bundle.py --domain 2 --v 0.1 --t 30.0 --epochs 10 --n-xyz 12 --pretrain-trials 0
```

Then verify:

```bash
python verify_outputs.py --base domain_2_v_0p1_t_30p0
```

If that passes, the public repo is in good shape for a cloud-based smoke test.
