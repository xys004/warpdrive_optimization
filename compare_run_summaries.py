import argparse
from pathlib import Path

import numpy as np

from postprocess_plots import SUCCESS_KEYS, compute_plane_map, load_run


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def _print_row(label: str, a: float, b: float) -> None:
    delta = b - a if np.isfinite(a) and np.isfinite(b) else np.nan
    print(f"{label:20s}  A={_fmt(a):>12s}  B={_fmt(b):>12s}  delta={_fmt(delta):>12s}")


def _final_value(table: dict, key: str) -> float:
    if key not in table:
        return np.nan
    values = np.asarray(table[key], dtype=float)
    return float(values[-1]) if values.size else np.nan


def _load_csv_table(path: Path) -> dict:
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return {name: np.asarray(arr[name], dtype=float) for name in arr.dtype.names}


def _load_legacy_run(base: str) -> dict:
    stem = Path(base)
    files = {
        "parameters": stem.parent / f"{stem.name}_parameters.csv",
        "losses": stem.parent / f"{stem.name}_losses.csv",
        "success_rates": stem.parent / f"{stem.name}_success_rates.csv",
        "violations": stem.parent / f"{stem.name}_violations.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing required legacy files for base '{stem}':\n{joined}")

    parameters = _load_csv_table(files["parameters"])
    losses = _load_csv_table(files["losses"])
    success_rates = _load_csv_table(files["success_rates"])
    violations = _load_csv_table(files["violations"])

    final_params = {
        "A": _final_value(parameters, "A"),
        "B": _final_value(parameters, "B"),
        "R0": _final_value(parameters, "R0"),
        "alpha": np.nan,
    }
    return {
        "scheme": "legacy",
        "base": str(stem),
        "final_params": final_params,
        "parameters": parameters,
        "losses": losses,
        "success_rates": success_rates,
        "violations": violations,
    }


def load_any_run(base: str) -> dict:
    stem = Path(base)
    revised_files = [
        stem.parent / f"{stem.name}_final_params.csv",
        stem.parent / f"{stem.name}_parameters.csv",
        stem.parent / f"{stem.name}_losses.csv",
        stem.parent / f"{stem.name}_success_rates.csv",
        stem.parent / f"{stem.name}_metadata.json",
    ]
    if all(path.exists() for path in revised_files):
        run = load_run(str(stem))
        run["scheme"] = "revised"
        run["base"] = str(stem)
        return run
    return _load_legacy_run(str(stem))


def compare_scalar_summaries(run_a: dict, run_b: dict) -> None:
    print("\nFinal parameters")
    for key in ("A", "B", "R0", "alpha"):
        a = float(run_a["final_params"].get(key, np.nan))
        b = float(run_b["final_params"].get(key, np.nan))
        _print_row(key, a, b)

    print("\nFinal success fractions / diagnostics")
    success_keys = sorted(set(run_a["success_rates"].keys()) | set(run_b["success_rates"].keys()))
    success_keys = [key for key in success_keys if key != "epoch"]
    for key in success_keys:
        a = _final_value(run_a["success_rates"], key)
        b = _final_value(run_b["success_rates"], key)
        _print_row(key, a, b)

    print("\nFinal loss components")
    loss_keys = (
        "total_loss",
        "physics_loss",
        "param_penalty",
        "reg_breach",
        "inv_alpha",
        "shear_pen",
        "E_soft_%",
    )
    for key in loss_keys:
        if key not in run_a["losses"] and key not in run_b["losses"]:
            continue
        a = _final_value(run_a["losses"], key)
        b = _final_value(run_b["losses"], key)
        _print_row(key, a, b)

    if "violations" in run_a and "violations" in run_b:
        print("\nLegacy violation counts")
        _print_row("violations", _final_value(run_a["violations"], "violations"), _final_value(run_b["violations"], "violations"))


def _field_summary(data: np.ndarray) -> tuple[float, float, float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(np.min(finite)),
        float(np.max(finite)),
        float(np.mean(finite)),
        float(np.max(np.abs(finite))),
    )


def compare_plane_summaries(run_a: dict, run_b: dict, planes, plane_n: int) -> None:
    if run_a["scheme"] != "revised" or run_b["scheme"] != "revised":
        print("\nPlane summaries skipped: direct XY/XZ field comparisons require revised bundles with metadata.json.")
        return

    for plane in planes:
        plane_a = compute_plane_map(run_a, plane=plane, plane_n=plane_n)
        plane_b = compute_plane_map(run_b, plane=plane, plane_n=plane_n)
        print(f"\nPlane summaries ({plane}, plane_n={plane_n})")
        for field in ("rho", "WEC_min", "NEC_min", "DEC_margin", "SEC"):
            a_min, a_max, a_mean, a_absmax = _field_summary(plane_a["fields"][field])
            b_min, b_max, b_mean, b_absmax = _field_summary(plane_b["fields"][field])
            print(field)
            print(f"  A: min={_fmt(a_min)} max={_fmt(a_max)} mean={_fmt(a_mean)} absmax={_fmt(a_absmax)}")
            print(f"  B: min={_fmt(b_min)} max={_fmt(b_max)} mean={_fmt(b_mean)} absmax={_fmt(b_absmax)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two legacy and/or revised run bundles numerically.")
    parser.add_argument("--base-a", required=True, help="First run base")
    parser.add_argument("--base-b", required=True, help="Second run base")
    parser.add_argument("--planes", nargs="*", default=["XY", "XZ"], help="Planes to summarize for revised bundles")
    parser.add_argument("--plane-n", type=int, default=300, help="Plane sampling density for field summaries")
    args = parser.parse_args()

    run_a = load_any_run(args.base_a)
    run_b = load_any_run(args.base_b)

    print(f"Comparing:\n  A = {args.base_a} ({run_a['scheme']})\n  B = {args.base_b} ({run_b['scheme']})")
    compare_scalar_summaries(run_a, run_b)
    compare_plane_summaries(run_a, run_b, planes=args.planes, plane_n=int(args.plane_n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
