"""Run the fixed development batch with reproducible, domain-stable seeds.

For constant-velocity runs, changing the time label should not by itself create a
physically distinct comoving diagnostic map. The batch therefore keeps the same
seed for the paired t=0.2 / t=30.0 runs inside each domain, so time comparisons
do not silently conflate physics with different random initializations.
"""

import argparse
import os
import shutil
import subprocess
import sys

DEFAULT_OUTDIR = "runs"
SRC = "einstein_optimizer.py"
CASES = [
    {"domain": 1, "v": 0.1, "t": 0.2, "seed": 147},
    {"domain": 1, "v": 0.1, "t": 30.0, "seed": 147},
    {"domain": 2, "v": 0.1, "t": 0.2, "seed": 247},
    {"domain": 2, "v": 0.1, "t": 30.0, "seed": 247},
]


def fmt_num(x: float) -> str:
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p") if "." in s else f"{s}p0"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fixed four-case development batch.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Directory where run artifacts are written.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing bundles in the output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    dst = os.path.join(outdir, "einstein_optimizer.py")
    if not os.path.exists(SRC):
        raise SystemExit("Cannot find einstein_optimizer.py in the current directory.")
    shutil.copy2(SRC, dst)

    py = sys.executable
    for case in CASES:
        domain = case["domain"]
        v = case["v"]
        t = case["t"]
        seed = case["seed"]
        print("\n" + "=" * 88)
        print(f"Launching: domain={domain}  v={v}  t={t}  seed={seed}")
        command = [py, "einstein_optimizer.py", "--domain", str(domain), "--v", str(v), "--t", str(t), "--seed", str(seed)]
        if args.overwrite:
            command.append("--overwrite")
        result = subprocess.run(
            command,
            cwd=outdir,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(f"Run failed for domain={domain}, v={v}, t={t}")

    print("\nAll runs completed. Output files live in:", os.path.abspath(outdir))
    print("You should see names like:")
    for case in CASES:
        print(f"  domain_{case['domain']}_v_{fmt_num(case['v'])}_t_{fmt_num(case['t'])}_*")


if __name__ == "__main__":
    main()
