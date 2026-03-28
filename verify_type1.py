#!/usr/bin/env python3
"""Hawking-Ellis Type I verification for the warp-drive optimization pipeline.

This script verifies that the stress-energy tensor T^a_b for the optimized
warp-bubble configurations is Hawking-Ellis Type I, which guarantees that
the principal-stress energy-condition margins are valid for ALL timelike
and null observers, not just the comoving diagnostic frame.

Theoretical basis:
- For the uniform-translation zero-vorticity class, K_ij is independent of
  the constant boost velocity v_i (only spatial derivatives of beta contribute).
- Therefore the ADM momentum constraint G^perp_i is unchanged from the static
  seed, where it vanishes by spherical symmetry.
- This gives q_i = 0 in the orthonormal comoving frame, making T^a_b block-diagonal.
- A block-diagonal T^a_b with real symmetric P_ij is automatically Hawking-Ellis
  Type I (diagonalizable with one timelike + three spacelike eigenvectors).
- For Type I, the standard energy conditions reduce to algebraic inequalities
  among the eigenvalues {-rho, lambda_1, lambda_2, lambda_3}.

Usage:
    python verify_type1.py --domain 1 --params_csv path/to/final_params.csv
    python verify_type1.py --domain 2 --A 1.848 --B 1.135 --R0 1.292
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import tensorflow as tf

# Reuse the pipeline's own physics and optimizer modules
from physics_core import (
    assemble_pressure_eigenvalues,
    principal_stress_margins,
    hawking_ellis_type1_diagnostic,
)
from einstein_optimizer import EinsteinTrainerCPU, CONFIG


def run_verification(domain_type, A, B, R0, N_xyz=48, v=0.1, t=30.0,
                     xyz_range=(-6.0, 6.0), output_dir=None):
    """Run full Type I verification on a single configuration."""

    trainer = EinsteinTrainerCPU(
        N_xyz=N_xyz,
        N_t=1,
        xyz_range=xyz_range,
        t_range=(t, t),
        domain_type=domain_type,
        alpha_mode="train",
        alpha_init=CONFIG["ALPHA_INIT"],
        alpha_floor_start=CONFIG["ALPHA_FLOOR_START"],
        alpha_floor_end=CONFIG["ALPHA_FLOOR_END"],
        alpha_warmup_frac=CONFIG["ALPHA_WARMUP_FRAC"],
        smoothing_target_pct=CONFIG["SMOOTHING_TARGET_PCT"],
        physics_scale=CONFIG["PHYSICS_SCALE"],
        L_breach=CONFIG["L_BREACH"],
        L_inv_alpha=CONFIG["L_INV_ALPHA"],
        velocity=CONFIG["VELOCITY"],
        L_shear=CONFIG["L_SHEAR"],
        use_dec_loss=CONFIG.get("USE_DEC_LOSS", True),
        dec_loss_weight=CONFIG.get("DEC_LOSS_WEIGHT", 0.25),
        v_mode="constant",
        v_coeffs=(0.0, v),
    )

    A_tf = tf.constant(A, dtype=tf.float32)
    B_tf = tf.constant(B, dtype=tf.float32)
    R0_tf = tf.constant(R0 if domain_type == 2 else 0.0, dtype=tf.float32)

    # Compute stress-energy components
    rho, Px, Py, Pz, Txy, Txz, Tyz, r = trainer.compute_components_on_coords(
        trainer.X, trainer.Y, trainer.Z, trainer.T, A_tf, B_tf, R0_tf
    )

    # Compute momentum flux (should be ~0)
    qx, qy, qz = trainer.compute_momentum_flux(
        trainer.X, trainer.Y, trainer.Z, trainer.T, A_tf, B_tf, R0_tf
    )

    # Type I diagnostic
    is_type1, flux_norm, eig_gap, type1_frac = hawking_ellis_type1_diagnostic(
        rho, Px, Py, Pz, Txy, Txz, Tyz, qx, qy, qz
    )

    # Energy condition margins
    nec_m, wec_m, dec_m, sec_m = principal_stress_margins(
        rho, Px, Py, Pz, Txy, Txz, Tyz
    )

    # Apply domain mask
    hard, _, _ = trainer.hard_masks(r, A_tf, B_tf, R0_tf)
    mask_bool = hard

    # Collect statistics
    def masked_stats(tensor, mask_bool):
        vals = tf.boolean_mask(tensor, mask_bool)
        return {
            "min": float(tf.reduce_min(vals).numpy()),
            "max": float(tf.reduce_max(vals).numpy()),
            "mean": float(tf.reduce_mean(vals).numpy()),
            "std": float(tf.math.reduce_std(vals).numpy()),
        }

    results = {
        "domain_type": domain_type,
        "parameters": {"A": A, "B": B, "R0": R0},
        "grid": {"N_xyz": N_xyz, "range": list(xyz_range)},
        "kinematics": {"v": v, "t": t},
        "type1_verification": {
            "type1_fraction_total": float(type1_frac.numpy()),
            "type1_fraction_masked": float(
                tf.reduce_mean(
                    tf.cast(tf.boolean_mask(is_type1, mask_bool), tf.float32)
                ).numpy()
            ),
            "flux_norm": masked_stats(flux_norm, mask_bool),
            "eigenvalue_gap": masked_stats(eig_gap, mask_bool),
        },
        "energy_conditions_masked": {
            "NEC": {
                "success_fraction": float(
                    tf.reduce_mean(
                        tf.cast(tf.boolean_mask(nec_m, mask_bool) >= -1e-9, tf.float32)
                    ).numpy()
                ),
                "margin": masked_stats(nec_m, mask_bool),
            },
            "WEC": {
                "success_fraction": float(
                    tf.reduce_mean(
                        tf.cast(tf.boolean_mask(wec_m, mask_bool) >= -1e-9, tf.float32)
                    ).numpy()
                ),
                "margin": masked_stats(wec_m, mask_bool),
            },
            "DEC": {
                "success_fraction": float(
                    tf.reduce_mean(
                        tf.cast(tf.boolean_mask(dec_m, mask_bool) >= -1e-9, tf.float32)
                    ).numpy()
                ),
                "margin": masked_stats(dec_m, mask_bool),
            },
            "SEC": {
                "success_fraction": float(
                    tf.reduce_mean(
                        tf.cast(tf.boolean_mask(sec_m, mask_bool) >= -1e-9, tf.float32)
                    ).numpy()
                ),
                "margin": masked_stats(sec_m, mask_bool),
            },
        },
    }

    # Print summary
    print("=" * 70)
    print("HAWKING-ELLIS TYPE I VERIFICATION")
    print(f"Domain {domain_type} | A={A:.4f}, B={B:.4f}, R0={R0:.4f}")
    print(f"Grid: {N_xyz}^3 | v={v}, t={t}")
    print("=" * 70)
    t1 = results["type1_verification"]
    print(f"\n  Type I fraction (full grid):   {t1['type1_fraction_total']:.6f}")
    print(f"  Type I fraction (masked shell): {t1['type1_fraction_masked']:.6f}")
    print(f"  Max |q| (flux norm):            {t1['flux_norm']['max']:.2e}")
    print(f"  Mean |q|:                       {t1['flux_norm']['mean']:.2e}")
    print(f"  Min eigenvalue gap:             {t1['eigenvalue_gap']['min']:.2e}")

    ec = results["energy_conditions_masked"]
    print("\n  Energy condition success fractions (all-observer, via Type I):")
    for name in ["NEC", "WEC", "DEC", "SEC"]:
        sf = ec[name]["success_fraction"]
        mn = ec[name]["margin"]["min"]
        print(f"    {name}: {sf:.4f}  (min margin: {mn:.4e})")

    is_type1_str = "TYPE I" if t1["type1_fraction_masked"] > 0.99 else "NOT fully Type I"
    print(f"\n  CONCLUSION: T^a_b is {is_type1_str}")
    if t1["type1_fraction_masked"] > 0.99:
        print("  => Principal-stress margins are VALID FOR ALL OBSERVERS")
    print("=" * 70)

    # Export
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"type1_verification_d{domain_type}.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {output_dir}/type1_verification_d{domain_type}.json")

    return results


def main():
    parser = argparse.ArgumentParser(description="Hawking-Ellis Type I verification")
    parser.add_argument("--domain", type=int, required=True, choices=[1, 2])
    parser.add_argument("--A", type=float, default=None)
    parser.add_argument("--B", type=float, default=None)
    parser.add_argument("--R0", type=float, default=0.0)
    parser.add_argument("--params_csv", type=str, default=None)
    parser.add_argument("--N_xyz", type=int, default=48)
    parser.add_argument("--v", type=float, default=0.1)
    parser.add_argument("--t", type=float, default=30.0)
    parser.add_argument("--output_dir", type=str, default="verification_output")
    args = parser.parse_args()

    if args.params_csv:
        with open(args.params_csv) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            A = float(row.get("A", row.get("a")))
            B = float(row.get("B", row.get("b")))
            R0 = float(row.get("R0", row.get("R_0", row.get("r0", 0.0))))
    else:
        A, B, R0 = args.A, args.B, args.R0

    run_verification(args.domain, A, B, R0,
                     N_xyz=args.N_xyz, v=args.v, t=args.t,
                     output_dir=args.output_dir)


if __name__ == "__main__":
    main()
