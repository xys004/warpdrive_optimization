import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path


def fmt_num(x: float) -> str:
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p") if "." in s else f"{s}p0"


def run_step(command, cwd: Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_optimizer(args, outdir: Path, base: str) -> None:
    """Run the optimizer stage with CLI overrides applied to the shared CONFIG."""
    from einstein_optimizer import CONFIG, run_cpu
    import numpy as np
    import tensorflow as tf

    cfg = copy.deepcopy(CONFIG)
    cfg["DOMAIN_TYPE"] = int(args.domain)
    cfg["VELOCITY"] = float(args.v)
    cfg["T_RANGE"] = (float(args.t), float(args.t))
    cfg["N_T"] = 1
    cfg["SHOW_PLOTS"] = False

    if args.seed is not None:
        cfg["SEED"] = int(args.seed)
    if args.epochs is not None:
        cfg["NUM_EPOCHS"] = int(args.epochs)
    if args.n_xyz is not None:
        cfg["N_XYZ"] = int(args.n_xyz)
    if args.pretrain_trials is not None:
        cfg["PRETRAIN_TRIALS"] = int(args.pretrain_trials)
    if args.pretrain_epochs is not None:
        cfg["PRETRAIN_EPOCHS"] = int(args.pretrain_epochs)
    cfg["OVERWRITE_OUTPUTS"] = bool(args.overwrite)

    tf.random.set_seed(cfg["SEED"])
    np.random.seed(cfg["SEED"])

    # The optimizer writes bundle files into the current working directory, so the
    # wrapper temporarily switches into outdir rather than re-implementing path logic.
    old_cwd = Path.cwd()
    os.chdir(outdir)
    try:
        print(
            f"Optimizer config: epochs={cfg['NUM_EPOCHS']} N_xyz={cfg['N_XYZ']} "
            f"pretrain_trials={cfg['PRETRAIN_TRIALS']} pretrain_epochs={cfg['PRETRAIN_EPOCHS']}"
        )
        run_cpu(cfg)
    finally:
        os.chdir(old_cwd)

    expected_base = f"domain_{args.domain}_v_{fmt_num(args.v)}_t_{fmt_num(args.t)}"
    if expected_base != base:
        raise RuntimeError(f"Internal base-name mismatch: expected {expected_base}, got {base}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate one optimizer run bundle, post-process plots, and verify outputs."
    )
    parser.add_argument("--domain", type=int, choices=[1, 2], required=True, help="Shell topology: 1=single shell, 2=double shell")
    parser.add_argument("--v", type=float, required=True, help="Constant bubble speed")
    parser.add_argument("--t", type=float, required=True, help="Time label for the run bundle")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed override")
    parser.add_argument("--epochs", type=int, default=None, help="Optional NUM_EPOCHS override for the optimizer")
    parser.add_argument("--n-xyz", type=int, default=None, help="Optional N_XYZ override for the optimizer grid")
    parser.add_argument("--pretrain-trials", type=int, default=None, help="Optional PRETRAIN_TRIALS override")
    parser.add_argument("--pretrain-epochs", type=int, default=None, help="Optional PRETRAIN_EPOCHS override")
    parser.add_argument("--outdir", default=".", help="Directory where the run bundle and plots are written")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing run bundle with the same basename")
    parser.add_argument("--with-colorbar", dest="with_colorbar", action="store_true", default=True, help="Include colorbars in field maps")
    parser.add_argument("--no-colorbar", dest="with_colorbar", action="store_false", help="Disable colorbars in field maps")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    py = Path(sys.executable)
    base = f"domain_{args.domain}_v_{fmt_num(args.v)}_t_{fmt_num(args.t)}"

    print(f"[1/3] Optimizer run: {base}")
    run_optimizer(args, outdir=outdir, base=base)

    post_cmd = [
        str(py),
        str((Path(__file__).resolve().parent / "postprocess_plots.py")),
        "--base",
        base,
        "--outdir",
        str(outdir),
    ]
    post_cmd.append("--with-colorbar" if args.with_colorbar else "--no-colorbar")

    verify_cmd = [
        str(py),
        str((Path(__file__).resolve().parent / "verify_outputs.py")),
        "--base",
        base,
    ]

    print(f"[2/3] Post-processing plots: {base}")
    run_step(post_cmd, cwd=outdir)

    print(f"[3/3] Verifying outputs: {base}")
    run_step(verify_cmd, cwd=outdir)

    print(f"OK: generated and verified run bundle {base} in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

