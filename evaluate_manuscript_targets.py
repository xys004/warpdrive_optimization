from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from einstein_optimizer import (
    CONFIG,
    EinsteinTrainerCPU,
    F32,
    build_run_metadata,
    ensure_output_paths_available,
)
from manuscript_targets import get_target


TARGET_CASES = {
    "domain_1_v_0p1_t_30p0": {"domain": 1, "v": 0.1, "t": 30.0},
    "domain_2_v_0p1_t_30p0": {"domain": 2, "v": 0.1, "t": 30.0},
}


def resolve_case(base_name: str) -> dict:
    """Return the manuscript target and physical case definition for a known basename.

    The manuscript targets currently exist only for the two published `t=30.0`,
    `v=0.1` benchmark cases. Keeping the mapping explicit avoids hidden parsing
    logic and makes the audit workflow easier to review.
    """
    target = get_target(base_name)
    if target is None:
        raise KeyError(f"No manuscript target is defined for {base_name}")
    if base_name not in TARGET_CASES:
        raise KeyError(f"No case definition is registered for {base_name}")
    case = copy.deepcopy(TARGET_CASES[base_name])
    case["target"] = copy.deepcopy(target)
    return case


def evaluation_alpha(cfg: dict) -> float:
    """Return the effective alpha used for fixed-parameter manuscript evaluation.

    The manuscript targets specify the geometric seed parameters `(A, B, R0)` but do
    not publish a separate `alpha`. For referee-facing audits we therefore evaluate
    the fields using the end-of-run mask sharpness implied by the configured alpha
    schedule, without performing any optimization updates.
    """
    if str(cfg.get("ALPHA_MODE", "train")) == "fixed":
        return float(cfg["ALPHA_INIT"])
    floor_end = float(cfg["ALPHA_FLOOR_END"])
    if str(cfg.get("ALPHA_MODE", "train")) == "schedule":
        return floor_end
    return float(cfg["ALPHA_INIT"] + floor_end)


def build_single_row_history(trainer: EinsteinTrainerCPU, cfg: dict, alpha_value: float) -> tuple[dict, dict, str]:
    """Evaluate one fixed parameter snapshot and package it into bundle-like history rows.

    The returned history has a single synthetic epoch so that downstream tooling can
    reuse the same CSV schema and plotting logic that the optimization pipeline uses.
    This keeps manuscript-target evaluation directly comparable to an optimized run.
    """
    target = cfg["MANUSCRIPT_TARGET"]
    trainer.create_variables(A_init=target["A"], B_init=target["B"], R0_init=max(target.get("R0", 0.0), 1e-4))
    trainer.set_from_values(target["A"], target["B"], target.get("R0"))

    A, B, R0 = trainer.get_ABR()
    alpha = tf.constant(float(alpha_value), dtype=tf.float32)
    rho, Px, Py, Pz, Txy, Txz, Tyz, r = trainer.compute_components(A, B, R0)
    w = trainer.soft_mask(r, A, B, R0, alpha)

    use_dec = bool(cfg.get("USE_DEC_LOSS", True))
    dec_weight = float(cfg.get("DEC_LOSS_WEIGHT", 0.25))
    L_phys = trainer.physics_loss_all_observers(rho, Px, Py, Pz, Txy, Txz, Tyz, w, use_dec=use_dec, dec_w=dec_weight)
    if trainer.L_shear > 0.0:
        shear_pen = trainer.L_shear * tf.reduce_mean(w * (Txy * Txy + Txz * Txz + Tyz * Tyz))
    else:
        shear_pen = F32(0.0)

    FN, FP_L, FP_R, E_tot = trainer.smoothing_error_continuous(r, A, B, R0, alpha)
    breach = tf.nn.relu(E_tot - F32(trainer.smoothing_target))
    L_smooth = F32(trainer.L_breach) * (breach ** 2)
    L_alpha = F32(trainer.L_inv_alpha) / (alpha + F32(1e-6))
    L_total = L_phys + L_smooth + L_alpha + shear_pen

    succ = trainer.success_rates_eig(rho, Px, Py, Pz, Txy, Txz, Tyz, r, A, B, R0)
    nec_margin, wec_margin, dec_margin, sec_margin = trainer.energy_margins_all_observers(Px, Py, Pz, Txy, Txz, Tyz, rho)

    history = {
        "loss_total": [float(L_total.numpy())],
        "loss_phys": [float(L_phys.numpy())],
        "loss_smooth": [float(L_smooth.numpy())],
        "loss_invalpha": [float(L_alpha.numpy())],
        "loss_shear": [float(shear_pen.numpy())],
        "alpha": [float(alpha.numpy())],
        "A": [float(A.numpy())],
        "B": [float(B.numpy())],
        "R0": [float(R0.numpy()) if R0 is not None else np.nan],
        "E_FN%": [float(FN.numpy() * 100.0)],
        "E_FP_L%": [float(FP_L.numpy() * 100.0)],
        "E_FP_R%": [float(FP_R.numpy() * 100.0)],
        "E_tot%": [float(E_tot.numpy() * 100.0)],
        "succ_nec": [float(succ["nec"].numpy())],
        "succ_wec": [float(succ["wec"].numpy())],
        "succ_dec": [float(succ["dec"].numpy())],
        "succ_sec": [float(succ["sec"].numpy())],
        "nec_margin_soft": [float(tf.reduce_mean(w * nec_margin).numpy())],
        "wec_margin_soft": [float(tf.reduce_mean(w * wec_margin).numpy())],
        "dec_margin_soft": [float(tf.reduce_mean(w * dec_margin).numpy())],
        "sec_margin_soft": [float(tf.reduce_mean(w * sec_margin).numpy())],
    }

    report_text, final_params = trainer.final_report(history)
    return history, final_params, report_text


def write_final_params(path: Path, final_params: dict) -> None:
    """Persist the one-row final parameter table used by the rest of the pipeline."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["A", "B", "R0", "alpha"])
        writer.writerow([
            final_params["A"],
            final_params["B"],
            "" if final_params["R0"] is None else final_params["R0"],
            final_params["alpha"],
        ])


def build_cfg(base_name: str, seed: int | None, overwrite: bool, alpha_override: float | None) -> dict:
    """Construct the evaluation configuration for a manuscript-target case."""
    case = resolve_case(base_name)
    cfg = copy.deepcopy(CONFIG)
    cfg["DOMAIN_TYPE"] = int(case["domain"])
    cfg["VELOCITY"] = float(case["v"])
    cfg["T_RANGE"] = (float(case["t"]), float(case["t"]))
    cfg["N_T"] = 1
    cfg["SEED"] = int(cfg["SEED"] if seed is None else seed)
    cfg["OVERWRITE_OUTPUTS"] = bool(overwrite)
    cfg["MANUSCRIPT_TARGET"] = case["target"]
    cfg["EVALUATION_ALPHA"] = float(alpha_override) if alpha_override is not None else evaluation_alpha(cfg)
    return cfg


def run_evaluation(base_name: str, outdir: Path, seed: int | None, overwrite: bool, alpha_override: float | None, with_colorbar: bool) -> int:
    """Generate a synthetic bundle and rerender plots for a manuscript target.

    This entry point intentionally mirrors the optimization bundle workflow: it writes
    CSV/JSON artifacts first, then runs the standard post-processing and verification
    scripts. The resulting bundle can therefore be compared directly against optimized
    runs without any special-case plotting code.
    """
    cfg = build_cfg(base_name, seed=seed, overwrite=overwrite, alpha_override=alpha_override)
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    try:
        os.chdir(outdir)
        ensure_output_paths_available(base_name, overwrite=overwrite)
        tf.random.set_seed(cfg["SEED"])
        np.random.seed(cfg["SEED"])

        trainer = EinsteinTrainerCPU(
            N_xyz=cfg["N_XYZ"],
            N_t=cfg["N_T"],
            xyz_range=cfg["XYZ_RANGE"],
            t_range=cfg["T_RANGE"],
            domain_type=cfg["DOMAIN_TYPE"],
            alpha_mode=cfg["ALPHA_MODE"],
            alpha_init=cfg["ALPHA_INIT"],
            alpha_floor_start=cfg["ALPHA_FLOOR_START"],
            alpha_floor_end=cfg["ALPHA_FLOOR_END"],
            alpha_warmup_frac=cfg["ALPHA_WARMUP_FRAC"],
            smoothing_target_pct=cfg["SMOOTHING_TARGET_PCT"],
            physics_scale=cfg["PHYSICS_SCALE"],
            L_breach=cfg["L_BREACH"],
            L_inv_alpha=cfg["L_INV_ALPHA"],
            velocity=cfg["VELOCITY"],
            L_shear=cfg["L_SHEAR"],
            use_dec_loss=cfg.get("USE_DEC_LOSS", True),
            dec_loss_weight=cfg.get("DEC_LOSS_WEIGHT", 0.25),
            v_mode=cfg.get("V_MODE", "linear"),
            v_coeffs=cfg.get("V_COEFFS", (0.0, 0.1)),
        )

        history, final_params, report_text = build_single_row_history(
            trainer, cfg, alpha_value=cfg["EVALUATION_ALPHA"]
        )
        trainer.export_history_csvs(history, base_name)
        write_final_params(Path(f"{base_name}_final_params.csv"), final_params)
        Path(f"{base_name}_final_report.txt").write_text(report_text, encoding="utf-8")

        metadata = build_run_metadata(cfg, final_params, base_name)
        metadata["run_kind"] = "manuscript_target_evaluation"
        metadata["manuscript_target"] = copy.deepcopy(cfg["MANUSCRIPT_TARGET"])
        metadata["evaluation_alpha"] = float(cfg["EVALUATION_ALPHA"])
        Path(f"{base_name}_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

        postprocess_cmd = [
            sys.executable,
            str(old_cwd / "postprocess_plots.py"),
            "--base",
            base_name,
            "--outdir",
            ".",
        ]
        postprocess_cmd.append("--with-colorbar" if with_colorbar else "--no-colorbar")
        subprocess.run(postprocess_cmd, cwd=outdir, check=True)

        verify_cmd = [
            sys.executable,
            str(old_cwd / "verify_outputs.py"),
            "--base",
            base_name,
        ]
        subprocess.run(verify_cmd, cwd=outdir, check=True)
    finally:
        os.chdir(old_cwd)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a manuscript target case without optimization and export a comparable run bundle."
    )
    parser.add_argument("--base", required=True, help="Known manuscript target basename, e.g. domain_2_v_0p1_t_30p0")
    parser.add_argument("--outdir", default="manuscript_target_bundles", help="Directory in which to write the synthetic bundle")
    parser.add_argument("--seed", type=int, default=None, help="Optional metadata seed override; no optimization is performed")
    parser.add_argument("--alpha", type=float, default=None, help="Optional effective alpha used for soft-mask diagnostics")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing manuscript-target bundle")
    parser.add_argument("--with-colorbar", action="store_true", help="Keep colorbars in the rerendered field maps")
    parser.add_argument("--no-colorbar", action="store_true", help="Disable colorbars in the rerendered field maps")
    args = parser.parse_args()

    if args.with_colorbar and args.no_colorbar:
        raise SystemExit("Choose at most one of --with-colorbar / --no-colorbar")
    with_colorbar = bool(args.with_colorbar)
    if args.no_colorbar:
        with_colorbar = False

    try:
        return run_evaluation(
            base_name=args.base,
            outdir=Path(args.outdir),
            seed=args.seed,
            overwrite=bool(args.overwrite),
            alpha_override=args.alpha,
            with_colorbar=with_colorbar,
        )
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
