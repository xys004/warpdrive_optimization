import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from output_naming import field_plot_base


FIELD_EXPORTS = {
    "rho": {"filename": "rho", "title": "rho"},
    "WEC_min": {"filename": "WECmin", "title": "WEC_min"},
    "NEC_min": {"filename": "NECmin", "title": "NEC_min"},
    "DEC_margin": {"filename": "DECmargin", "title": "DEC_margin"},
    "SEC": {"filename": "SEC", "title": "SEC"},
}

PLANE_NAMES = ("XY", "XZ")
SUCCESS_KEYS = ("nec", "wec", "dec", "sec")
REQUIRED_METADATA_KEYS = (
    "domain_type",
    "v_mode",
    "velocity",
    "time",
    "N_xyz",
    "xyz_range",
)
REQUIRED_LOSS_COLUMNS = (
    "epoch",
    "total_loss",
    "physics_loss",
    "reg_breach",
    "inv_alpha",
)
REQUIRED_PARAM_COLUMNS = ("epoch", "A", "B", "alpha")
REQUIRED_SUCCESS_COLUMNS = ("epoch", "nec", "wec", "dec", "sec")
FINAL_PARAM_KEYS = ("A", "B", "alpha")
SOFT_MARGIN_COLUMNS = (
    "nec_margin_soft",
    "wec_margin_soft",
    "dec_margin_soft",
    "sec_margin_soft",
)


def configure_style():
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.grid": False,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.2,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _read_numeric_table(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                value = row[name].strip()
                columns[name].append(np.nan if value == "" else float(value))
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def _read_single_row(path):
    table = _read_numeric_table(path)
    if not table:
        raise ValueError(f"CSV file is empty: {path}")
    n_rows = len(next(iter(table.values())))
    if n_rows != 1:
        raise ValueError(f"Expected exactly one data row in {path}, found {n_rows}")
    return {name: values[0] for name, values in table.items()}


def _resolve_base(base):
    base_path = Path(base)
    if base_path.suffix:
        raise ValueError("--base must be a basename/stem without file suffix")
    return base_path


def _require_columns(table, required, label):
    missing = [name for name in required if name not in table]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{label} is missing required columns: {joined}")


def _require_metadata(metadata):
    missing = [name for name in REQUIRED_METADATA_KEYS if name not in metadata]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"metadata.json is missing required keys: {joined}")


def _validate_epochs(run):
    epochs_losses = np.asarray(run["losses"]["epoch"], dtype=float)
    epochs_params = np.asarray(run["parameters"]["epoch"], dtype=float)
    epochs_success = np.asarray(run["success_rates"]["epoch"], dtype=float)
    if len(epochs_losses) == 0:
        raise ValueError("losses.csv contains no epochs")
    if len(epochs_losses) != len(epochs_params) or len(epochs_losses) != len(epochs_success):
        raise ValueError("losses, parameters, and success_rates must have the same number of epochs")
    if not np.allclose(epochs_losses, epochs_params, equal_nan=False):
        raise ValueError("Epoch columns differ between losses.csv and parameters.csv")
    if not np.allclose(epochs_losses, epochs_success, equal_nan=False):
        raise ValueError("Epoch columns differ between losses.csv and success_rates.csv")


def _validate_final_params(run):
    metadata = run["metadata"]
    final_params = run["final_params"]
    last_params = run["parameters"]
    meta_final = metadata.get("final_parameters", {})

    for key in FINAL_PARAM_KEYS:
        if key not in final_params:
            raise ValueError(f"final_params.csv is missing '{key}'")
        if key not in last_params:
            raise ValueError(f"parameters.csv is missing '{key}'")

    domain_type = int(metadata["domain_type"])
    if domain_type == 2:
        if "R0" not in final_params or "R0" not in last_params:
            raise ValueError("Domain 2 requires R0 in final_params.csv and parameters.csv")
    elif domain_type != 1:
        raise ValueError(f"Unsupported domain_type={domain_type}; expected 1 or 2")

    for key in FINAL_PARAM_KEYS:
        last_value = float(last_params[key][-1])
        final_value = float(final_params[key])
        if not np.isfinite(last_value) or not np.isfinite(final_value):
            raise ValueError(f"Non-finite final parameter detected for {key}")
        if not np.isclose(last_value, final_value, rtol=1e-5, atol=1e-7):
            warnings.warn(
                f"Final parameter mismatch for {key}: parameters.csv last epoch={last_value}, final_params.csv={final_value}",
                stacklevel=2,
            )
        if key in meta_final:
            meta_value = float(meta_final[key])
            if not np.isclose(meta_value, final_value, rtol=1e-5, atol=1e-7):
                warnings.warn(
                    f"Metadata final_parameters[{key}]={meta_value} differs from final_params.csv={final_value}",
                    stacklevel=2,
                )

    if domain_type == 2:
        last_r0 = float(last_params["R0"][-1])
        final_r0 = float(final_params["R0"])
        if not np.isfinite(last_r0) or not np.isfinite(final_r0):
            raise ValueError("Domain 2 requires finite R0 values")
        if not np.isclose(last_r0, final_r0, rtol=1e-5, atol=1e-7):
            warnings.warn(
                f"Final parameter mismatch for R0: parameters.csv last epoch={last_r0}, final_params.csv={final_r0}",
                stacklevel=2,
            )
    elif np.isfinite(final_params.get("R0", np.nan)):
        warnings.warn("Domain 1 run includes an unused finite R0 in final_params.csv; it will be ignored.", stacklevel=2)


def validate_run(run):
    metadata = run["metadata"]
    _require_metadata(metadata)
    _require_columns(run["losses"], REQUIRED_LOSS_COLUMNS, "losses.csv")
    _require_columns(run["parameters"], REQUIRED_PARAM_COLUMNS, "parameters.csv")
    _require_columns(run["success_rates"], REQUIRED_SUCCESS_COLUMNS, "success_rates.csv")
    _validate_epochs(run)
    _validate_final_params(run)

    domain_type = int(metadata["domain_type"])
    if domain_type == 2 and "R0" not in run["parameters"]:
        raise ValueError("Domain 2 requires an R0 column in parameters.csv")

    xyz_range = metadata["xyz_range"]
    if not isinstance(xyz_range, list) or len(xyz_range) != 2:
        raise ValueError("metadata.json must provide xyz_range as a two-element list")
    xyz_min, xyz_max = float(xyz_range[0]), float(xyz_range[1])
    if not xyz_max > xyz_min:
        raise ValueError("metadata.json must satisfy xyz_range[1] > xyz_range[0]")

    n_xyz = int(metadata["N_xyz"])
    if n_xyz < 2:
        raise ValueError("metadata.json must satisfy N_xyz >= 2")

    if str(metadata.get("v_mode", "constant")) != "constant":
        velocity_settings = metadata.get("velocity_settings")
        if not isinstance(velocity_settings, dict):
            raise ValueError("Linear-mode metadata must include a velocity_settings object")
        v_coeffs = velocity_settings.get("v_coeffs")
        if not isinstance(v_coeffs, list) or len(v_coeffs) != 2:
            raise ValueError("Linear-mode metadata must include velocity_settings.v_coeffs as a two-element list")

    for key in SUCCESS_KEYS:
        values = np.asarray(run["success_rates"][key], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and (np.nanmin(finite) < -1e-6 or np.nanmax(finite) > 100.0 + 1e-6):
            warnings.warn(f"Success fraction '{key}' falls outside [0, 100]%. Check exported diagnostics.", stacklevel=2)

    if any(name not in run["success_rates"] for name in SOFT_MARGIN_COLUMNS):
        warnings.warn(
            "success_rates.csv is missing one or more soft-margin columns; plotted success fractions still use nec/wec/dec/sec.",
            stacklevel=2,
        )


def load_run(base):
    base_path = _resolve_base(base)
    stem = base_path.name
    root = base_path.parent if str(base_path.parent) not in ("", ".") else Path(".")

    files = {
        "final_params": root / f"{stem}_final_params.csv",
        "parameters": root / f"{stem}_parameters.csv",
        "losses": root / f"{stem}_losses.csv",
        "success_rates": root / f"{stem}_success_rates.csv",
        "metadata": root / f"{stem}_metadata.json",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing required input files for base '{stem}':\n{joined}")

    with files["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    run = {
        "base_path": base_path,
        "base_name": stem,
        "root": root,
        "paths": files,
        "metadata": metadata,
        "final_params": _read_single_row(files["final_params"]),
        "parameters": _read_numeric_table(files["parameters"]),
        "losses": _read_numeric_table(files["losses"]),
        "success_rates": _read_numeric_table(files["success_rates"]),
    }
    validate_run(run)
    return run


def _build_trainer(metadata, diagnostic_n_xyz=None, build_grid=True):
    from einstein_optimizer import CONFIG, EinsteinTrainerCPU

    alpha_cfg = metadata.get("alpha_settings", {})
    velocity_cfg = metadata.get("velocity_settings", {})
    velocity = float(velocity_cfg.get("snapshot_velocity", metadata["velocity"]))
    v_mode = str(metadata.get("v_mode", "constant"))
    # Rebuild the exact kinematic law recorded by the optimizer bundle. Falling back
    # to module-level defaults would make linear-mode rerenders non-reproducible.
    if isinstance(velocity_cfg.get("v_coeffs"), list) and len(velocity_cfg["v_coeffs"]) == 2:
        v_coeffs = tuple(float(v) for v in velocity_cfg["v_coeffs"])
    else:
        v_coeffs = tuple(CONFIG.get("V_COEFFS", (0.0, 0.1)))

    # For the paper's uniformly translated class, diagnostics are interpreted in the
    # orthonormal bubble-centered/comoving frame. When v is constant and the shell
    # topology is fixed, changing time only changes the lab-frame displacement; it does
    # not create a physically distinct comoving field map.
    effective_v_mode = "constant" if v_mode == "constant" else v_mode

    physics_cfg = metadata.get("physics_settings", {})

    trainer = EinsteinTrainerCPU(
        N_xyz=int(diagnostic_n_xyz if diagnostic_n_xyz is not None else metadata["N_xyz"]),
        N_t=1,
        xyz_range=tuple(metadata["xyz_range"]),
        t_range=(float(metadata["time"]), float(metadata["time"])),
        domain_type=int(metadata["domain_type"]),
        alpha_mode=str(alpha_cfg.get("mode", CONFIG["ALPHA_MODE"])),
        alpha_init=float(alpha_cfg.get("init", CONFIG["ALPHA_INIT"])),
        alpha_floor_start=float(alpha_cfg.get("floor_start", CONFIG["ALPHA_FLOOR_START"])),
        alpha_floor_end=float(alpha_cfg.get("floor_end", CONFIG["ALPHA_FLOOR_END"])),
        alpha_warmup_frac=float(alpha_cfg.get("warmup_frac", CONFIG["ALPHA_WARMUP_FRAC"])),
        smoothing_target_pct=float(physics_cfg.get("smoothing_target_pct", CONFIG["SMOOTHING_TARGET_PCT"])),
        physics_scale=float(physics_cfg.get("physics_scale", CONFIG["PHYSICS_SCALE"])),
        L_breach=float(physics_cfg.get("L_breach", CONFIG["L_BREACH"])),
        L_inv_alpha=float(physics_cfg.get("L_inv_alpha", CONFIG["L_INV_ALPHA"])),
        velocity=velocity,
        L_shear=float(physics_cfg.get("L_shear", CONFIG["L_SHEAR"])),
        use_dec_loss=bool(physics_cfg.get("use_dec_loss", CONFIG.get("USE_DEC_LOSS", True))),
        dec_loss_weight=float(physics_cfg.get("dec_loss_weight", CONFIG.get("DEC_LOSS_WEIGHT", 0.25))),
        build_grid=build_grid,
        v_mode=effective_v_mode,
        v_coeffs=v_coeffs,
    )
    return trainer


def _as_3d(values, n_xyz):
    return np.asarray(values, dtype=float).reshape((n_xyz, n_xyz, n_xyz))


def _sample_plane_axes(metadata, plane, plane_n=None):
    xyz_min, xyz_max = [float(v) for v in metadata["xyz_range"]]
    n = int(plane_n if plane_n is not None else metadata["N_xyz"])
    axis = np.linspace(xyz_min, xyz_max, n, dtype=float)
    plane = plane.upper()
    if plane == "XY":
        X, Y = np.meshgrid(axis, axis, indexing="ij")
        Z = np.zeros_like(X)
        return axis, axis, X, Y, Z, "x", "y"
    if plane == "XZ":
        X, Z = np.meshgrid(axis, axis, indexing="ij")
        Y = np.zeros_like(X)
        return axis, axis, X, Y, Z, "x", "z"
    raise ValueError(f"Unsupported plane '{plane}'. Use one of {PLANE_NAMES}.")


def compute_plane_map(run, plane, interface_buffer=0.0, plane_n=None):
    metadata = run["metadata"]
    params = run["final_params"]
    trainer = _build_trainer(metadata, diagnostic_n_xyz=plane_n, build_grid=False)
    trainer.create_variables(
        A_init=float(params["A"]),
        B_init=float(params["B"]),
        R0_init=float(params["R0"]) if np.isfinite(params.get("R0", np.nan)) else 1.0,
    )
    trainer.set_from_values(
        A_val=float(params["A"]),
        B_val=float(params["B"]),
        R0_val=float(params["R0"]) if np.isfinite(params.get("R0", np.nan)) else None,
    )

    A, B, R0 = trainer.get_ABR()
    axis_x, axis_y, X, Y, Z, xlabel, ylabel = _sample_plane_axes(metadata, plane, plane_n=plane_n)
    T = np.full_like(X, float(metadata["time"]), dtype=float)

    rho, Px, Py, Pz, Txy, Txz, Tyz, r = trainer.compute_components_on_coords(
        X.reshape(-1),
        Y.reshape(-1),
        Z.reshape(-1),
        T.reshape(-1),
        A,
        B,
        R0,
    )

    from physics_core import principal_stress_margins

    nec_margin, wec_margin, dec_margin, sec_margin = principal_stress_margins(
        rho, Px, Py, Pz, Txy, Txz, Tyz
    )
    hard_mask_tf, _, _ = trainer.hard_masks(r, A, B, R0)

    n = len(axis_x)
    r_2d = np.asarray(r.numpy(), dtype=float).reshape((n, n))
    valid_mask = np.asarray(hard_mask_tf.numpy(), dtype=bool).reshape((n, n))

    r_inner = float(R0.numpy()) if int(metadata["domain_type"]) == 2 else float((2.0 / B).numpy())
    r_outer = float((2.0 / B).numpy()) if int(metadata["domain_type"]) == 2 else float(trainer.r_cap.numpy())
    if interface_buffer > 0.0:
        valid_mask &= r_2d >= (r_inner + interface_buffer)
        valid_mask &= r_2d <= (r_outer - interface_buffer)

    raw_fields = {
        "rho": np.asarray(rho.numpy(), dtype=float).reshape((n, n)),
        "WEC_min": np.asarray(wec_margin.numpy(), dtype=float).reshape((n, n)),
        "NEC_min": np.asarray(nec_margin.numpy(), dtype=float).reshape((n, n)),
        "DEC_margin": np.asarray(dec_margin.numpy(), dtype=float).reshape((n, n)),
        "SEC": np.asarray(sec_margin.numpy(), dtype=float).reshape((n, n)),
    }

    finite_mask = np.ones_like(valid_mask, dtype=bool)
    for values in raw_fields.values():
        finite_mask &= np.isfinite(values)
    valid_mask &= finite_mask

    masked_fields = {
        name: np.where(valid_mask, values, np.nan) for name, values in raw_fields.items()
    }

    return {
        "plane": plane.upper(),
        "axis_x": axis_x,
        "axis_y": axis_y,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "fields": masked_fields,
        "parameters": {
            "A": float(A.numpy()),
            "B": float(B.numpy()),
            "R0": None if R0 is None else float(R0.numpy()),
            "alpha": float(params.get("alpha", np.nan)),
            "r_inner": r_inner,
            "r_outer": r_outer,
        },
    }


def compute_maps(run, interface_buffer=0.0, diagnostic_n_xyz=None):
    metadata = run["metadata"]
    params = run["final_params"]
    trainer = _build_trainer(metadata, diagnostic_n_xyz=diagnostic_n_xyz)
    trainer.create_variables(
        A_init=float(params["A"]),
        B_init=float(params["B"]),
        R0_init=float(params["R0"]) if np.isfinite(params.get("R0", np.nan)) else 1.0,
    )
    trainer.set_from_values(
        A_val=float(params["A"]),
        B_val=float(params["B"]),
        R0_val=float(params["R0"]) if np.isfinite(params.get("R0", np.nan)) else None,
    )

    A, B, R0 = trainer.get_ABR()
    rho, Px, Py, Pz, Txy, Txz, Tyz, r = trainer.compute_components(A, B, R0)

    # Field maps are rebuilt from the same shared principal-stress eigenvalue diagnostics used by
    # the optimizer and by success_rates.csv: diagonalize P_ij in the comoving frame and form
    # rho, WEC_min, NEC_min, DEC_margin, and SEC from the resulting principal stresses.
    from physics_core import principal_stress_margins

    nec_margin, wec_margin, dec_margin, sec_margin = principal_stress_margins(
        rho, Px, Py, Pz, Txy, Txz, Tyz
    )
    hard_mask_tf, _, _ = trainer.hard_masks(r, A, B, R0)

    n_xyz = int(trainer.N_xyz)
    x_vals = np.asarray(trainer.x_vals.numpy(), dtype=float)
    y_vals = np.asarray(trainer.y_vals.numpy(), dtype=float)
    z_vals = np.asarray(trainer.z_vals.numpy(), dtype=float)
    r_3d = _as_3d(r.numpy(), n_xyz)
    hard_mask = np.asarray(hard_mask_tf.numpy(), dtype=bool).reshape((n_xyz, n_xyz, n_xyz))

    r_inner = float(R0.numpy()) if int(metadata["domain_type"]) == 2 else float((2.0 / B).numpy())
    r_outer = float((2.0 / B).numpy()) if int(metadata["domain_type"]) == 2 else float(trainer.r_cap.numpy())

    valid_mask = hard_mask.copy()
    if interface_buffer > 0.0:
        valid_mask &= r_3d >= (r_inner + interface_buffer)
        valid_mask &= r_3d <= (r_outer - interface_buffer)

    raw_fields = {
        "rho": _as_3d(rho.numpy(), n_xyz),
        "WEC_min": _as_3d(wec_margin.numpy(), n_xyz),
        "NEC_min": _as_3d(nec_margin.numpy(), n_xyz),
        "DEC_margin": _as_3d(dec_margin.numpy(), n_xyz),
        "SEC": _as_3d(sec_margin.numpy(), n_xyz),
    }

    finite_mask = np.ones_like(valid_mask, dtype=bool)
    for values in raw_fields.values():
        finite_mask &= np.isfinite(values)
    valid_mask &= finite_mask

    masked_fields = {
        name: np.where(valid_mask, values, np.nan) for name, values in raw_fields.items()
    }

    return {
        "trainer": trainer,
        "x": x_vals,
        "y": y_vals,
        "z": z_vals,
        "r": r_3d,
        "valid_mask": valid_mask,
        "fields": masked_fields,
        "parameters": {
            "A": float(A.numpy()),
            "B": float(B.numpy()),
            "R0": None if R0 is None else float(R0.numpy()),
            "alpha": float(params.get("alpha", np.nan)),
            "r_inner": r_inner,
            "r_outer": r_outer,
        },
    }


def _extract_plane(volume, plane, x_vals, y_vals, z_vals):
    plane = plane.upper()
    if plane == "XY":
        z_idx = int(np.argmin(np.abs(z_vals)))
        return volume[:, :, z_idx], x_vals, y_vals, "x", "y"
    if plane == "XZ":
        y_idx = int(np.argmin(np.abs(y_vals)))
        return volume[:, y_idx, :], x_vals, z_vals, "x", "z"
    raise ValueError(f"Unsupported plane '{plane}'. Use one of {PLANE_NAMES}.")


def _symmetric_norm(data):
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    vmax = float(np.nanmax(np.abs(finite)))
    if vmax <= 0.0:
        vmax = 1.0
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def plot_map(
    data,
    axis_x,
    axis_y,
    xlabel,
    ylabel,
    title,
    output_path,
    with_colorbar=True,
    figsize=(6.5, 5.8),
    dpi=300,
    interpolation="nearest",
    xlim=None,
    ylim=None,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap("bwr").copy()
    cmap.set_bad(color="white")
    ax.set_facecolor("white")

    image = ax.imshow(
        np.ma.masked_invalid(data).T,
        origin="lower",
        extent=(axis_x[0], axis_x[-1], axis_y[0], axis_y[-1]),
        cmap=cmap,
        norm=_symmetric_norm(data),
        interpolation=interpolation,
        aspect="equal",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if with_colorbar:
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=11)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _normalize_series(series):
    series = np.asarray(series, dtype=float)
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return np.full_like(series, np.nan, dtype=float)
    first = None
    for value in series:
        if np.isfinite(value) and abs(value) > 0.0:
            first = abs(value)
            break
    if first is None:
        peak = float(np.nanmax(np.abs(finite)))
        denom = peak if peak > 0.0 else 1.0
    else:
        denom = first
    return series / denom


def plot_losses(run, output_path, figsize=(8.5, 5.4), dpi=300):
    losses = run["losses"]
    epochs = losses["epoch"]
    keys = ["total_loss", "physics_loss", "reg_breach", "inv_alpha", "shear_pen"]
    labels = {
        "total_loss": "total",
        "physics_loss": "physics",
        "reg_breach": "reg_breach",
        "inv_alpha": "inv_alpha",
        "shear_pen": "shear_pen",
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for key in keys:
        if key not in losses:
            continue
        values = np.asarray(losses[key], dtype=float)
        if not np.any(np.isfinite(values)):
            continue
        if np.allclose(np.nan_to_num(values, nan=0.0), 0.0):
            continue
        normalized = np.abs(_normalize_series(values))
        normalized = np.clip(normalized, 1e-16, None)
        ax.semilogy(epochs, normalized, label=labels[key])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("normalized component")
    ax.set_title("loss components")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_success(run, output_path, figsize=(8.0, 5.2), dpi=300):
    success = run["success_rates"]
    epochs = success["epoch"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for key, label in (("nec", "NEC"), ("wec", "WEC"), ("dec", "DEC"), ("sec", "SEC")):
        if key in success:
            ax.plot(epochs, success[key], label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("success fraction [%]")
    ax.set_ylim(0.0, 100.0)
    ax.set_title("success fractions")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_params(run, output_path, figsize=(8.5, 6.2), dpi=300):
    params = run["parameters"]
    epochs = params["epoch"]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        height_ratios=(2.0, 1.0),
    )

    for key, label in (("A", "A"), ("B", "B"), ("R0", "R0")):
        if key in params:
            values = np.asarray(params[key], dtype=float)
            if np.any(np.isfinite(values)):
                ax_top.plot(epochs, values, label=label)
    ax_top.set_ylabel("shell parameters")
    ax_top.set_title("parameter evolution")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="best", frameon=False)

    if "alpha" in params:
        ax_bottom.plot(epochs, params["alpha"], color="black", label="alpha")
        ax_bottom.legend(loc="best", frameon=False)
    ax_bottom.set_xlabel("Epoch")
    ax_bottom.set_ylabel(r"$\alpha$")
    ax_bottom.grid(True, alpha=0.25)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _parse_planes(values):
    planes = []
    for value in values:
        for token in value.split(","):
            token = token.strip().upper()
            if token:
                planes.append(token)
    if not planes:
        raise ValueError("At least one plane must be requested")
    invalid = [plane for plane in planes if plane not in PLANE_NAMES]
    if invalid:
        joined = ", ".join(invalid)
        valid = ", ".join(PLANE_NAMES)
        raise ValueError(f"Unsupported plane(s): {joined}. Valid choices: {valid}")
    return tuple(dict.fromkeys(planes))


def main():
    parser = argparse.ArgumentParser(
        description="Post-process one optimizer run and export manuscript-ready PNG figures."
    )
    parser.add_argument("--base", required=True, help="Run basename, e.g. domain_2_v_0p1_t_30p0")
    parser.add_argument(
        "--planes",
        nargs="+",
        default=list(PLANE_NAMES),
        help="Plane list, e.g. --planes XY XZ or --planes XY,XZ",
    )
    parser.add_argument("--outdir", default=None, help="Directory for output PNG files")
    parser.add_argument(
        "--with-colorbar",
        dest="with_colorbar",
        action="store_true",
        default=True,
        help="Include colorbars on 2D map figures (default)",
    )
    parser.add_argument(
        "--no-colorbar",
        dest="with_colorbar",
        action="store_false",
        help="Disable colorbars on 2D map figures",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution")
    parser.add_argument(
        "--map-size",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(6.5, 5.8),
        help="Map figure size in inches",
    )
    parser.add_argument(
        "--line-size",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(8.5, 5.4),
        help="Line figure size in inches",
    )
    parser.add_argument(
        "--interface-buffer",
        type=float,
        default=0.0,
        help="Optional radial buffer removed from each shell boundary before plotting",
    )
    parser.add_argument(
        "--diagnostic-nxyz",
        type=int,
        default=None,
        help="Optional plane sampling density for post-processing only. Use this to recompute smoother XY/XZ diagnostic maps from the final parameters without rerunning the optimizer.",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        metavar=("XMIN", "XMAX"),
        default=None,
        help="Optional x-axis limits for field-map figures.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Optional y-axis limits for field-map figures. Applied to y on XY and z on XZ.",
    )
    parser.add_argument(
        "--interpolation",
        choices=["nearest", "bilinear", "bicubic"],
        default="nearest",
        help="Image interpolation used when rendering field maps. This affects display only, not the underlying diagnostics.",
    )
    args = parser.parse_args()

    configure_style()

    run = load_run(args.base)
    planes = _parse_planes(args.planes)
    outdir = Path(args.outdir) if args.outdir else run["root"]
    outdir.mkdir(parents=True, exist_ok=True)

    base_name = run["base_name"]
    field_base_name = field_plot_base(
        base_name=base_name,
        final_params=run["final_params"],
        domain_type=int(run["metadata"]["domain_type"]),
    )

    for plane in planes:
        plane_map = compute_plane_map(
            run,
            plane=plane,
            interface_buffer=float(args.interface_buffer),
            plane_n=args.diagnostic_nxyz,
        )
        for field_name, spec in FIELD_EXPORTS.items():
            output_path = outdir / f"{field_base_name}_{plane}_{spec['filename']}.png"
            plot_map(
                data=plane_map["fields"][field_name],
                axis_x=plane_map["axis_x"],
                axis_y=plane_map["axis_y"],
                xlabel=plane_map["xlabel"],
                ylabel=plane_map["ylabel"],
                title=f"{spec['title']} ({plane})",
                output_path=output_path,
                with_colorbar=args.with_colorbar,
                figsize=tuple(args.map_size),
                dpi=args.dpi,
                interpolation=args.interpolation,
                xlim=tuple(args.xlim) if args.xlim else None,
                ylim=tuple(args.ylim) if args.ylim else None,
            )

    plot_losses(
        run,
        outdir / f"{base_name}_loss_components.png",
        figsize=tuple(args.line_size),
        dpi=args.dpi,
    )
    plot_success(
        run,
        outdir / f"{base_name}_success_fractions.png",
        figsize=tuple(args.line_size),
        dpi=args.dpi,
    )
    plot_params(
        run,
        outdir / f"{base_name}_params.png",
        figsize=(args.line_size[0], max(args.line_size[1], 6.2)),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()




