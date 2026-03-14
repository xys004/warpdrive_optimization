import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from output_naming import expected_plot_paths

REQUIRED_INPUT_SUFFIXES = (
    "final_params.csv",
    "parameters.csv",
    "losses.csv",
    "success_rates.csv",
    "metadata.json",
)

EXPECTED_HEADERS = {
    "final_params": ("A", "B", "R0", "alpha"),
    "parameters_domain_1": ("epoch", "A", "B", "alpha"),
    "parameters_domain_2": ("epoch", "A", "B", "R0", "alpha"),
    "losses": ("epoch", "total_loss", "physics_loss", "reg_breach", "inv_alpha", "E_soft_%", "shear_pen"),
    "success_rates": (
        "epoch",
        "nec",
        "wec",
        "dec",
        "sec",
        "nec_margin_soft",
        "wec_margin_soft",
        "dec_margin_soft",
        "sec_margin_soft",
    ),
}

FINAL_KEYS = ("A", "B", "alpha")
SUCCESS_KEYS = ("nec", "wec", "dec", "sec")
LEGACY_SUCCESS_KEYS = ("h1", "h2", "h3", "h4", "h5", "h6")


@dataclass
class VerificationResult:
    errors: List[str]
    warnings: List[str]
    notes: List[str]

    def ok(self) -> bool:
        return not self.errors


def _resolve_base(base: str) -> Path:
    base_path = Path(base)
    if base_path.suffix:
        raise ValueError("--base must be a basename/stem without a file suffix")
    return base_path


def _build_paths(base: Path) -> Dict[str, Path]:
    stem = base.name
    root = base.parent if str(base.parent) not in ("", ".") else Path(".")
    return {
        "root": root,
        "final_params": root / f"{stem}_final_params.csv",
        "parameters": root / f"{stem}_parameters.csv",
        "losses": root / f"{stem}_losses.csv",
        "success_rates": root / f"{stem}_success_rates.csv",
        "metadata": root / f"{stem}_metadata.json",
    }


def _read_csv(path: Path) -> tuple[List[str], List[dict]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def _to_float(value: str) -> float:
    value = value.strip()
    return np.nan if value == "" else float(value)


def _row_as_float_map(row: dict) -> Dict[str, float]:
    return {key: _to_float(value) for key, value in row.items()}


def _check_exact_header(actual: Sequence[str], expected: Sequence[str], label: str, errors: List[str]) -> None:
    if tuple(actual) != tuple(expected):
        errors.append(f"{label} header mismatch: expected {list(expected)}, found {list(actual)}")


def _check_required_files(paths: Dict[str, Path], errors: List[str]) -> None:
    for key in ("final_params", "parameters", "losses", "success_rates", "metadata"):
        if not paths[key].exists():
            errors.append(f"Missing required file: {paths[key]}")


def _load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _check_success_schema(headers: Sequence[str], errors: List[str]) -> None:
    legacy_found = [name for name in LEGACY_SUCCESS_KEYS if name in headers]
    if legacy_found:
        errors.append(f"success_rates.csv uses legacy columns {legacy_found}; expected nec,wec,dec,sec")
    missing = [name for name in SUCCESS_KEYS if name not in headers]
    if missing:
        errors.append(f"success_rates.csv is missing revised success columns: {missing}")


def _check_final_parameter_consistency(
    metadata: dict,
    final_params_rows: List[dict],
    parameters_rows: List[dict],
    domain_type: int,
    tol: float,
    errors: List[str],
    warnings: List[str],
) -> None:
    if len(final_params_rows) != 1:
        errors.append(f"final_params.csv must contain exactly one data row; found {len(final_params_rows)}")
        return
    if not parameters_rows:
        errors.append("parameters.csv contains no data rows")
        return

    final_row = _row_as_float_map(final_params_rows[0])
    last_row = _row_as_float_map(parameters_rows[-1])
    metadata_final = metadata.get("final_parameters")
    if not isinstance(metadata_final, dict):
        errors.append("metadata.json is missing final_parameters")
        return

    keys = list(FINAL_KEYS)
    if domain_type == 2:
        keys.append("R0")

    for key in keys:
        if key not in final_row:
            errors.append(f"final_params.csv is missing {key}")
            continue
        if key not in last_row:
            errors.append(f"parameters.csv is missing {key}")
            continue
        if key not in metadata_final:
            warnings.append(f"metadata.json final_parameters is missing {key}")
            continue

        values = {
            "final_params.csv": final_row[key],
            "parameters.csv last row": last_row[key],
            "metadata.json": float(metadata_final[key]) if metadata_final[key] is not None else np.nan,
        }
        finite_values = [v for v in values.values() if np.isfinite(v)]
        if not finite_values:
            errors.append(f"All final values for {key} are non-finite")
            continue
        ref = finite_values[0]
        for label, value in values.items():
            if not np.isfinite(value):
                errors.append(f"Non-finite final value for {key} in {label}")
                continue
            if not np.isclose(value, ref, rtol=tol, atol=tol):
                errors.append(f"Final {key} mismatch: {label}={value} is inconsistent with reference {ref}")


def _check_plot_files(paths: Dict[str, Path], errors: List[str], notes: List[str]) -> None:
    missing = [path for path in paths["plots"] if not path.exists()]
    if missing:
        errors.append("Missing expected plot files:")
        errors.extend([f"  {path}" for path in missing])
        return
    notes.append(f"Verified {len(paths['plots'])} expected plot files")


def _check_timestamps(paths: Dict[str, Path], notes: List[str], warnings: List[str]) -> None:
    inputs = [paths[key] for key in ("final_params", "parameters", "losses", "success_rates", "metadata")]
    plots = list(paths["plots"])
    if not all(path.exists() for path in inputs + plots):
        return

    newest_input = max(inputs, key=lambda path: path.stat().st_mtime)
    oldest_plot = min(plots, key=lambda path: path.stat().st_mtime)
    newest_plot = max(plots, key=lambda path: path.stat().st_mtime)

    notes.append(
        "Input timestamps: newest input is "
        f"{newest_input.name} @ {newest_input.stat().st_mtime:.0f}"
    )
    notes.append(
        "Plot timestamps: oldest/newest plot are "
        f"{oldest_plot.name} @ {oldest_plot.stat().st_mtime:.0f} and {newest_plot.name} @ {newest_plot.stat().st_mtime:.0f}"
    )
    if oldest_plot.stat().st_mtime + 1.0 < newest_input.stat().st_mtime:
        warnings.append("Some plot files are older than the newest run artifact; plots may be stale")


def verify(base: str, tol: float = 1e-5) -> VerificationResult:
    errors: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []

    base_path = _resolve_base(base)
    paths = _build_paths(base_path)
    _check_required_files(paths, errors)
    if errors:
        return VerificationResult(errors=errors, warnings=warnings, notes=notes)

    metadata = _load_metadata(paths["metadata"])
    domain_type = int(metadata.get("domain_type", -1))
    if domain_type not in (1, 2):
        errors.append(f"metadata.json has invalid domain_type={domain_type}; expected 1 or 2")
        return VerificationResult(errors=errors, warnings=warnings, notes=notes)

    # Linear-mode runs need extra metadata to remain reproducible. Constant-velocity
    # runs can be reconstructed from the snapshot velocity alone.
    if str(metadata.get("v_mode", "constant")) != "constant":
        velocity_settings = metadata.get("velocity_settings")
        if not isinstance(velocity_settings, dict):
            errors.append("metadata.json is missing velocity_settings for a linear-mode run")
        else:
            v_coeffs = velocity_settings.get("v_coeffs")
            if not isinstance(v_coeffs, list) or len(v_coeffs) != 2:
                errors.append("metadata.json must provide velocity_settings.v_coeffs for a linear-mode run")

    final_headers, final_rows = _read_csv(paths["final_params"])
    param_headers, param_rows = _read_csv(paths["parameters"])
    loss_headers, loss_rows = _read_csv(paths["losses"])
    success_headers, success_rows = _read_csv(paths["success_rates"])

    _check_exact_header(final_headers, EXPECTED_HEADERS["final_params"], "final_params.csv", errors)
    expected_param_header = EXPECTED_HEADERS[f"parameters_domain_{domain_type}"]
    _check_exact_header(param_headers, expected_param_header, "parameters.csv", errors)
    _check_exact_header(loss_headers, EXPECTED_HEADERS["losses"], "losses.csv", errors)
    _check_exact_header(success_headers, EXPECTED_HEADERS["success_rates"], "success_rates.csv", errors)
    _check_success_schema(success_headers, errors)

    if len(loss_rows) != len(param_rows) or len(loss_rows) != len(success_rows):
        errors.append(
            "CSV row-count mismatch: losses.csv, parameters.csv, and success_rates.csv must contain the same number of epochs"
        )

    if loss_rows and param_rows and success_rows:
        loss_epochs = np.asarray([_to_float(row["epoch"]) for row in loss_rows], dtype=float)
        param_epochs = np.asarray([_to_float(row["epoch"]) for row in param_rows], dtype=float)
        success_epochs = np.asarray([_to_float(row["epoch"]) for row in success_rows], dtype=float)
        if not np.allclose(loss_epochs, param_epochs, equal_nan=False):
            errors.append("Epoch mismatch between losses.csv and parameters.csv")
        if not np.allclose(loss_epochs, success_epochs, equal_nan=False):
            errors.append("Epoch mismatch between losses.csv and success_rates.csv")

    _check_final_parameter_consistency(
        metadata=metadata,
        final_params_rows=final_rows,
        parameters_rows=param_rows,
        domain_type=domain_type,
        tol=tol,
        errors=errors,
        warnings=warnings,
    )
    final_param_map = _row_as_float_map(final_rows[0]) if final_rows else {}
    paths["plots"] = expected_plot_paths(paths["root"], base_path.name, final_param_map, domain_type)
    _check_plot_files(paths, errors, notes)
    _check_timestamps(paths, notes, warnings)

    notes.append(f"Verified run bundle for {base_path.name} in {paths['root'].resolve()}")
    notes.append(f"Detected domain_type={domain_type} with {len(param_rows)} optimization epochs")

    return VerificationResult(errors=errors, warnings=warnings, notes=notes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lightweight reproducibility and consistency checker for one completed run bundle."
    )
    parser.add_argument("--base", required=True, help="Run basename, e.g. domain_2_v_0p1_t_30p0")
    parser.add_argument("--tol", type=float, default=1e-5, help="Absolute/relative tolerance for numeric consistency checks")
    args = parser.parse_args()

    try:
        result = verify(args.base, tol=float(args.tol))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    if result.errors:
        print("FAIL")
        for message in result.errors:
            print(f"- {message}")
        for message in result.warnings:
            print(f"warning: {message}")
        return 1

    print("OK")
    for message in result.notes:
        print(f"- {message}")
    for message in result.warnings:
        print(f"warning: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


