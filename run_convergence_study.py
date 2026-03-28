#!/usr/bin/env python3
"""Convergence study: verify that energy-condition diagnostics are stable across grid resolutions.

Addresses referee concern about grid sensitivity by running the Type I verification
and energy-condition diagnostics at N_xyz = {48, 64, 96} and comparing results.
"""
import json
import os
from verify_type1 import run_verification

RESOLUTIONS = [48, 64, 96]
OUTPUT_DIR = "convergence_study"

# Manuscript parameters
CONFIGS = [
    {"domain": 1, "A": 0.970, "B": 1.321, "R0": 0.0, "label": "single_shell"},
    {"domain": 2, "A": 1.848, "B": 1.135, "R0": 1.292, "label": "double_shell"},
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    for cfg in CONFIGS:
        label = cfg["label"]
        all_results[label] = {}
        print(f"\n{'#'*70}")
        print(f"# {label.upper()}: A={cfg['A']}, B={cfg['B']}, R0={cfg['R0']}")
        print(f"{'#'*70}")

        for N in RESOLUTIONS:
            print(f"\n--- Resolution N_xyz = {N} ---")
            res = run_verification(
                cfg["domain"], cfg["A"], cfg["B"], cfg["R0"],
                N_xyz=N, v=0.1, t=30.0,
                output_dir=os.path.join(OUTPUT_DIR, f"{label}_N{N}")
            )
            all_results[label][f"N{N}"] = res

    # Summary comparison table
    print("\n" + "=" * 80)
    print("CONVERGENCE SUMMARY")
    print("=" * 80)
    for label in all_results:
        print(f"\n{label}:")
        print(f"  {'N_xyz':>6} | {'Type1%':>8} | {'NEC%':>8} | {'WEC%':>8} | {'DEC%':>8} | {'SEC%':>8} | {'max|q|':>10}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
        for N in RESOLUTIONS:
            r = all_results[label][f"N{N}"]
            t1 = r["type1_verification"]["type1_fraction_masked"]
            ec = r["energy_conditions_masked"]
            mq = r["type1_verification"]["flux_norm"]["max"]
            print(f"  {N:>6} | {t1:>8.4f} | {ec['NEC']['success_fraction']:>8.4f} | "
                  f"{ec['WEC']['success_fraction']:>8.4f} | {ec['DEC']['success_fraction']:>8.4f} | "
                  f"{ec['SEC']['success_fraction']:>8.4f} | {mq:>10.2e}")

    with open(os.path.join(OUTPUT_DIR, "convergence_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {OUTPUT_DIR}/convergence_summary.json")


if __name__ == "__main__":
    main()
