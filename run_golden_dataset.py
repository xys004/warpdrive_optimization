"""Generate the small referee-facing Phase B golden dataset.

The script wraps the lower-level optimizer and plotter CLIs with a stable manifest
of cases defined in golden_dataset_cases.py. It is meant to answer a very specific
question: "What are the exact runs we want a referee or future collaborator to
reproduce first?"

The workflow is deliberately conservative:

1. run the optimizer bundle generator for each manifest case
2. immediately rerender the maps using the case-specific high-resolution settings
3. re-run output verification after the rerender

This gives us one compact command for the Phase B dataset while keeping the
underlying optimizer and plotter scripts independently usable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from golden_dataset_cases import CASE_GROUPS, get_case_group
from output_naming import fmt_num


def build_base_name(case: dict) -> str:
    """Return the optimizer basename for a manifest case."""

    optimizer = case["optimizer"]
    return f"domain_{optimizer['domain']}_v_{fmt_num(optimizer['v'])}_t_{fmt_num(optimizer['t'])}"


def run_step(command: list[str], cwd: Path) -> None:
    """Execute one subprocess step and fail fast on non-zero exit status."""

    result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode != 0:
        joined = " ".join(command)
        raise SystemExit(f"Command failed with exit code {result.returncode}: {joined}")


def generate_case(case: dict, outdir: Path, overwrite: bool) -> None:
    """Generate one golden-dataset case end to end."""

    optimizer = case["optimizer"]
    plotting = case["plotting"]
    base = build_base_name(case)
    repo_root = Path(__file__).resolve().parent
    py = sys.executable

    print("\n" + "=" * 88)
    print(f"Case: {case['id']}")
    print(f"Description: {case['description']}")
    print(f"Base: {base}")

    generate_cmd = [
        py,
        str(repo_root / "generate_run_bundle.py"),
        "--domain",
        str(optimizer["domain"]),
        "--v",
        str(optimizer["v"]),
        "--t",
        str(optimizer["t"]),
        "--seed",
        str(optimizer["seed"]),
        "--epochs",
        str(optimizer["epochs"]),
        "--n-xyz",
        str(optimizer["n_xyz"]),
        "--pretrain-trials",
        str(optimizer["pretrain_trials"]),
        "--pretrain-epochs",
        str(optimizer["pretrain_epochs"]),
        "--outdir",
        ".",
    ]
    if overwrite:
        generate_cmd.append("--overwrite")
    generate_cmd.append("--with-colorbar" if plotting["with_colorbar"] else "--no-colorbar")

    rerender_cmd = [
        py,
        str(repo_root / "postprocess_plots.py"),
        "--base",
        base,
        "--outdir",
        ".",
        "--diagnostic-nxyz",
        str(plotting["diagnostic_nxyz"]),
        "--dpi",
        str(plotting["dpi"]),
        "--map-size",
        str(plotting["map_size"][0]),
        str(plotting["map_size"][1]),
        "--line-size",
        str(plotting["line_size"][0]),
        str(plotting["line_size"][1]),
        "--interpolation",
        plotting["interpolation"],
        "--xlim",
        str(plotting["xlim"][0]),
        str(plotting["xlim"][1]),
        "--ylim",
        str(plotting["ylim"][0]),
        str(plotting["ylim"][1]),
    ]
    rerender_cmd.append("--with-colorbar" if plotting["with_colorbar"] else "--no-colorbar")

    verify_cmd = [
        py,
        str(repo_root / "verify_outputs.py"),
        "--base",
        base,
    ]

    print("[1/3] Generate optimizer bundle")
    run_step(generate_cmd, cwd=outdir)

    print("[2/3] Re-render with golden-dataset plotting settings")
    run_step(rerender_cmd, cwd=outdir)

    print("[3/3] Verify final outputs")
    run_step(verify_cmd, cwd=outdir)


def parse_args() -> argparse.Namespace:
    """Build the CLI for the Phase B dataset runner."""

    parser = argparse.ArgumentParser(description="Generate the Phase B referee-facing golden dataset.")
    parser.add_argument(
        "--group",
        choices=sorted(CASE_GROUPS),
        default="core",
        help="Manifest group to generate: core, time_audit, or all.",
    )
    parser.add_argument(
        "--outdir",
        default="golden_dataset",
        help="Directory where the generated bundles and plots are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing bundles inside the output directory.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the Phase B dataset runner."""

    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cases = get_case_group(args.group)
    print(f"Generating Phase B group '{args.group}' into {outdir}")
    for case in cases:
        generate_case(case, outdir=outdir, overwrite=bool(args.overwrite))

    print("\nOK: Phase B golden dataset completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
