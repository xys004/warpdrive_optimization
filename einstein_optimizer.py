import argparse
import copy
import json
import math
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm as _tqdm

from physics_core import assemble_pressure_eigenvalues, principal_stress_margins

warnings.filterwarnings("ignore")

CONFIG = {
    # Central configuration for the referee-facing single-snapshot workflow.
    "QUIET": True,
    "USE_TQDM": False,
    "SHOW_PLOTS": False,
    "PRINT_EVERY": 0,
    "V_MODE": "constant",
    "V_COEFFS": (0.0, 0.1),
    "VELOCITY": 1.0,
    "DOMAIN_TYPE": 2,
    "N_XYZ": 48,
    "XYZ_RANGE": (-6.0, 6.0),
    "N_T": 1,
    "T_RANGE": (30.0, 30.0),
    "NUM_EPOCHS": 2500,
    "LR": 3e-3,
    "SEED": 47,
    "ALPHA_MODE": "train",
    "ALPHA_INIT": 40.0,
    "ALPHA_FLOOR_START": 20.0,
    "ALPHA_FLOOR_END": 200.0,
    "ALPHA_WARMUP_FRAC": 1.0,
    "SMOOTHING_TARGET_PCT": 1.0,
    "PHYSICS_SCALE": 1e-12,
    "L_BREACH": 1e6,
    "L_INV_ALPHA": 1e3,
    "L_SHEAR": 0.0,
    "USE_DEC_LOSS": True,
    "DEC_LOSS_WEIGHT": 0.25,
    "OVERWRITE_OUTPUTS": False,
    "A_INIT": 1.0,
    "B_INIT": 1.0,
    "R0_INIT": 1.0,
    "FD_H": 1e-4,
    "R0_MARGIN": 0.02,
    "HARD_TOL": 1e-9,
    "RCAP_EPS": 1e-3,
    "PRETRAIN_TRIALS": 8,
    "PRETRAIN_EPOCHS": 60,
    "PRETRAIN_LR": 3e-3,
    "INIT_RANGES": {"A": (0.5, 2.0), "B": (0.5, 2.0), "R0": (0.3, 1.3)},
}

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.keras.backend.set_floatx("float32")
tf.random.set_seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])

QUIET = CONFIG.get("QUIET", False)
USE_TQDM = CONFIG.get("USE_TQDM", True)


def log(*args, **kwargs):
    if not QUIET:
        print(*args, **kwargs)


def tqdm_opt(*args, **kwargs):
    kwargs.setdefault("disable", not USE_TQDM)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("mininterval", 1.0)
    kwargs.setdefault("dynamic_ncols", True)
    return _tqdm(*args, **kwargs)


def F32(x):
    return tf.constant(float(x), tf.float32)


PI32 = F32(np.pi)
EPS = F32(1e-12)
FD_H = F32(CONFIG["FD_H"])
R0_MARGIN = F32(CONFIG["R0_MARGIN"])
HARD_TOL = F32(CONFIG["HARD_TOL"])
RCAP_EPS = F32(CONFIG["RCAP_EPS"])


def softplus_pos(u):
    u = tf.cast(u, tf.float32)
    return tf.nn.softplus(u) + F32(1e-6)


def inv_softplus_pos(y):
    y = max(float(y), 1e-6)
    return float(np.log(np.exp(y) - 1.0))


def fmt_num(x):
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p") if "." in s else f"{s}p0"


def current_snapshot_velocity(cfg):
    """Return the physical velocity associated with the exported snapshot.

    For constant-velocity runs this is just VELOCITY. For linear runs we record the
    instantaneous velocity at the exported snapshot time so downstream tools do not
    need to guess how the basename's `v` tag was obtained.
    """
    if str(cfg.get("V_MODE", "linear")) == "constant":
        return float(cfg["VELOCITY"])
    c0, c1 = cfg.get("V_COEFFS", (0.0, 0.1))
    return float(c0 + c1 * float(cfg["T_RANGE"][0]))


def build_base_name(cfg):
    """Build the canonical basename used by the paper pipeline.

    The basename intentionally stays compact because it appears in every artifact.
    Full reproducibility metadata is stored in metadata.json rather than encoded
    into filenames.
    """
    vtag = fmt_num(current_snapshot_velocity(cfg))
    ttag = fmt_num(float(cfg["T_RANGE"][0]))
    return f"domain_{cfg['DOMAIN_TYPE']}_v_{vtag}_t_{ttag}"


def optimizer_output_paths(base_name):
    """List the files written directly by the optimizer stage."""
    return [
        f"{base_name}_final_params.csv",
        f"{base_name}_parameters.csv",
        f"{base_name}_losses.csv",
        f"{base_name}_success_rates.csv",
        f"{base_name}_metadata.json",
        f"{base_name}_final_report.txt",
    ]


def ensure_output_paths_available(base_name, overwrite=False):
    """Fail early unless the caller explicitly allows overwriting an existing bundle.

    A silent overwrite makes referee-facing runs impossible to audit. We therefore
    stop before training begins unless the caller deliberately opts in.
    """
    existing = [path for path in optimizer_output_paths(base_name) if os.path.exists(path)]
    if existing and not overwrite:
        joined = "\n".join(existing)
        raise FileExistsError(
            "Refusing to overwrite an existing optimizer bundle. "
            "Use --overwrite to replace these files:\n" + joined
        )



class EinsteinTrainerCPU:
    """Single-snapshot optimizer used by the referee-facing paper pipeline.

    The trainer keeps the ansatz intentionally small: we optimize the analytic seed
    parameters `(A, B, R0)` and the mask sharpness `alpha`, while all physics-based
    diagnostics are derived from the resulting stress tensor. The class therefore
    acts as the central contract between the optimizer, the CSV exports, and the
    plotting/post-processing scripts.
    """

    def __init__(
        self,
        N_xyz,
        N_t,
        xyz_range,
        t_range,
        domain_type,
        alpha_mode,
        alpha_init,
        alpha_floor_start,
        alpha_floor_end,
        alpha_warmup_frac,
        smoothing_target_pct,
        physics_scale,
        L_breach,
        L_inv_alpha,
        velocity,
        L_shear,
        use_dec_loss,
        dec_loss_weight,
        v_mode="linear",
        v_coeffs=(0.0, 0.1),
        build_grid=True,
    ):
        self.N_xyz = int(N_xyz)
        self.N_t = int(N_t)
        self.xyz_min, self.xyz_max = [float(v) for v in xyz_range]
        self.t_min, self.t_max = [float(v) for v in t_range]
        self.domain_type = int(domain_type)
        self.alpha_mode = str(alpha_mode)
        self.alpha_init = float(alpha_init)
        self.alpha_floor_start = float(alpha_floor_start)
        self.alpha_floor_end = float(alpha_floor_end)
        self.alpha_warmup_frac = float(alpha_warmup_frac)
        self.smoothing_target = float(smoothing_target_pct) / 100.0
        self.physics_scale = float(physics_scale)
        self.L_breach = float(L_breach)
        self.L_inv_alpha = float(L_inv_alpha)
        self.L_shear = float(L_shear)
        self.use_dec_loss = bool(use_dec_loss)
        self.dec_loss_weight = float(dec_loss_weight)
        self.vel = float(velocity)
        self.v_mode = str(v_mode)
        c0, c1 = v_coeffs if v_coeffs is not None else (0.0, 0.0)
        self.v0 = float(c0)
        self.v1 = float(c1)

        # The public API still accepts N_t / t_range for compatibility with early
        # notebooks, but the revised workflow exports a single diagnostic snapshot.
        #
        # `build_grid=False` is used by high-resolution plotting utilities that only
        # need direct plane evaluations. Avoiding the full NxNxN mesh here prevents
        # large CPU-memory spikes during manuscript-quality rerenders.
        self.x_vals = tf.linspace(F32(self.xyz_min), F32(self.xyz_max), self.N_xyz)
        self.y_vals = tf.linspace(F32(self.xyz_min), F32(self.xyz_max), self.N_xyz)
        self.z_vals = tf.linspace(F32(self.xyz_min), F32(self.xyz_max), self.N_xyz)
        self.t_vals = tf.constant([self.t_min], tf.float32)
        self.N_t = 1
        self.grid_built = bool(build_grid)
        if self.grid_built:
            X, Y, Z, T = tf.meshgrid(
                self.x_vals, self.y_vals, self.z_vals, self.t_vals, indexing="ij"
            )
            self.X = tf.reshape(tf.cast(X, tf.float32), [-1])
            self.Y = tf.reshape(tf.cast(Y, tf.float32), [-1])
            self.Z = tf.reshape(tf.cast(Z, tf.float32), [-1])
            self.T = tf.reshape(tf.cast(T, tf.float32), [-1])
        else:
            self.X = tf.constant([], dtype=tf.float32)
            self.Y = tf.constant([], dtype=tf.float32)
            self.Z = tf.constant([], dtype=tf.float32)
            self.T = tf.constant([], dtype=tf.float32)

        rmax_box = np.sqrt(3.0) * max(abs(self.xyz_min), abs(self.xyz_max))
        self.r_cap = F32(rmax_box)

        self.A_raw = None
        self.B_raw = None
        self.R0_raw = None
        self.alpha_raw = None
        log("OPTIMIZER - CPU (EFE with explicit v; r=||x||; MC pre-training)")
        grid_label = f"{len(self.X)} pts" if self.grid_built else "direct-eval mode"
        log(
            f"Grid: {grid_label} | domain={self.domain_type} | "
            f"r_cap~{rmax_box:.4f} | v={self.vel}, t={self.t_min}"
        )

    @tf.function
    def V(self, t):
        t = tf.cast(t, tf.float32)
        if self.v_mode == "constant":
            return tf.ones_like(t) * F32(self.vel)
        return F32(self.v0) + F32(self.v1) * t

    @tf.function
    def r_geom(self, X, Y, Z):
        return tf.sqrt(tf.maximum(X * X + Y * Y + Z * Z, EPS))

    def physical_B_to_raw(self, B_value):
        """Convert a physical shell parameter `B` into the unconstrained optimization variable.

        Domain 1 uses a shifted parameterization `B = B_min + softplus(B_raw)` so that
        the hard shell always fits inside the finite plotting box. Callers that reason
        in terms of the physical `B` reported in CSV files should therefore use this
        helper instead of inverting `softplus` directly.
        """
        B_value = float(B_value)
        if self.domain_type == 1:
            B_min = 2.0 / (float(self.r_cap.numpy()) - float(RCAP_EPS.numpy()))
            B_value = max(B_value, B_min + 1e-6)
            return inv_softplus_pos(B_value - B_min)
        return inv_softplus_pos(B_value)

    def create_variables(self, A_init=1.0, B_init=1.0, R0_init=1.0):
        self.A_raw = tf.Variable(inv_softplus_pos(A_init), dtype=tf.float32)
        self.B_raw = tf.Variable(self.physical_B_to_raw(B_init), dtype=tf.float32)
        self.R0_raw = (
            tf.Variable(inv_softplus_pos(max(R0_init, 1e-4)), dtype=tf.float32)
            if self.domain_type == 2
            else None
        )
        self.alpha_raw = (
            tf.Variable(
                inv_softplus_pos(max(self.alpha_init - self.alpha_floor_start, 1e-3)),
                dtype=tf.float32,
            )
            if self.alpha_mode == "train"
            else None
        )
        A, B, R0 = self.get_ABR()
        if self.domain_type == 1:
            thr = float(2.0 / float(B.numpy()))
            log(
                f"Init: A={A.numpy():.4f} B={B.numpy():.4f} | "
                f"threshold 2/B={thr:.4f} | r_cap={float(self.r_cap.numpy()):.4f}"
            )
        else:
            log(f"Init: A={A.numpy():.4f} B={B.numpy():.4f} R0={R0.numpy():.4f}")

    def map_B(self, B_raw):
        if self.domain_type == 1:
            B_min = F32(2.0) / (self.r_cap - RCAP_EPS)
            return B_min + softplus_pos(B_raw)
        return softplus_pos(B_raw)

    def get_ABR(self):
        A = softplus_pos(self.A_raw)
        B = self.map_B(self.B_raw)
        R0 = self.map_R0(self.R0_raw, B) if self.domain_type == 2 else None
        return A, B, R0

    def set_from_values(self, A_val, B_val, R0_val=None):
        """Load physical parameter values directly into the trainable variables.

        This helper is used by pretraining, audits, and manuscript-target evaluation.
        It interprets `A`, `B`, and `R0` as the same physical quantities exported to
        CSV, rather than as raw unconstrained optimization coordinates.
        """
        self.A_raw.assign(inv_softplus_pos(A_val))
        self.B_raw.assign(self.physical_B_to_raw(B_val))
        if self.domain_type == 2 and R0_val is not None:
            _, B, _ = self.get_ABR()
            upper = float((2.0 / float(B.numpy())) - 2.0 * CONFIG["R0_MARGIN"])
            lo = CONFIG["R0_MARGIN"] + 1e-6
            hi = max(upper - 1e-6, lo + 1e-6)
            R0c = float(np.clip(R0_val, lo, hi))
            if upper > CONFIG["R0_MARGIN"]:
                s = (R0c - CONFIG["R0_MARGIN"]) / (upper - CONFIG["R0_MARGIN"])
            else:
                s = 0.5
            s = np.clip(s, 1e-6, 1.0 - 1e-6)
            self.R0_raw.assign(float(np.log(s / (1.0 - s))))

    def map_R0(self, R0_raw, B):
        B = tf.cast(B, tf.float32)
        R0_raw = tf.cast(R0_raw, tf.float32)
        upper = tf.maximum(F32(2.0) / B - F32(2.0) * R0_MARGIN, R0_MARGIN)
        s = tf.sigmoid(R0_raw)
        return R0_MARGIN + s * (upper - R0_MARGIN)

    def alpha_current(self, epoch, total_epochs):
        warmN = max(int(self.alpha_warmup_frac * total_epochs), 1)
        t = min(epoch, warmN) / float(warmN)
        floor_now = self.alpha_floor_start + t * (
            self.alpha_floor_end - self.alpha_floor_start
        )
        floor_now = F32(floor_now)
        if self.alpha_mode == "schedule":
            return floor_now
        if self.alpha_mode == "fixed":
            return F32(self.alpha_init)
        return softplus_pos(self.alpha_raw) + floor_now

    @tf.function
    def beta_r_piecewise(self, r, A, B, R0):
        """Evaluate the radial shift-vector seed ansatz beta(r) for both domain types.

        The shift vector beta(r) is the radial profile of the warp-bubble metric
        perturbation. Its square root form ensures beta >= 0 everywhere.

        Domain 1 (single shell):
            The ansatz is derived from integrating a Yukawa-type source. The active
            shell region is r >= 2/B (capped to the plotting box). Inside this
            threshold the shift is set to zero to avoid the coordinate singularity
            near r=0.

            bracket = 10*exp(-2) - exp(-Br)*(B^2*r^2 + 2Br + 2)
            beta_d1 = sqrt( (8*pi*A) / (B^3 * r) * max(bracket, 0) )  for r >= 2/B

        Domain 2 (double shell / annular region):
            The active shell is R0 <= r <= 2/B.  Inside R0 the shift is zero;
            outside 2/B the shift freezes at its boundary value (tail region).

            beta_mid = sqrt( (8*pi/r) * (A/B) * (1 - exp(-B*(r-R0))) )  for R0 <= r <= 2/B
            beta_hi  = same formula evaluated at r = 2/B (constant tail)  for r > 2/B

        Clipping to 1e10 prevents float32 overflow in rare degenerate configurations
        without altering the gradient landscape in physically reasonable parameter ranges.
        """
        r = tf.maximum(tf.cast(r, tf.float32), EPS)
        A = tf.maximum(tf.abs(tf.cast(A, tf.float32)), EPS)
        B = tf.maximum(tf.cast(B, tf.float32), EPS)
        Br = B * r

        # --- Domain 1: single-shell Yukawa-integrated ansatz ---
        bracket = F32(10.0) * tf.exp(-F32(2.0)) - tf.exp(-Br) * (
            B * B * r * r + F32(2.0) * B * r + F32(2.0)
        )
        factor = (F32(8.0) * PI32 * A) / tf.maximum(B**3 * r, EPS)
        beta_d1_raw = tf.sqrt(
            tf.clip_by_value(
                tf.maximum(factor * tf.maximum(bracket, EPS), EPS), EPS, F32(1e10)
            )
        )
        # Zero inside the threshold 2/B to avoid the near-origin singularity.
        beta_d1 = tf.where(r >= (F32(2.0) / B), beta_d1_raw, tf.zeros_like(beta_d1_raw))

        # --- Domain 2: double-shell / annular ansatz ---
        R = tf.maximum(
            tf.cast(R0 if R0 is not None else R0_MARGIN, tf.float32), R0_MARGIN
        )
        # Active mid-region: exponential rise from inner boundary R.
        mid_arg = tf.maximum(F32(1.0) - tf.exp(-B * (r - R)), EPS)
        beta_mid = tf.sqrt(
            tf.clip_by_value((F32(8.0) * PI32 / r) * (A / B) * mid_arg, EPS, F32(1e10))
        )
        # Tail region (r > 2/B): freeze at the outer-shell boundary value.
        tail_arg = tf.maximum(F32(1.0) - tf.exp(-B * ((F32(2.0) / B) - R)), EPS)
        beta_hi = tf.sqrt(
            tf.clip_by_value((F32(8.0) * PI32 / r) * (A / B) * tail_arg, EPS, F32(1e10))
        )
        beta_low = tf.zeros_like(beta_mid)
        beta_d2 = tf.where(r < R, beta_low, tf.where(r <= (F32(2.0) / B), beta_mid, beta_hi))

        return tf.where(
            tf.equal(F32(float(self.domain_type)), F32(1.0)), beta_d1, beta_d2
        )

    @tf.function
    def beta_and_derivs(self, r, A, B, R0):
        """Return beta(r) and its first two radial derivatives via central differences.

        The Einstein tensor components require d(beta)/dr and d^2(beta)/dr^2.
        Analytic differentiation of the piecewise ansatz is error-prone at the
        shell boundaries, so we use second-order central finite differences with
        step size FD_H = 1e-4 (set in CONFIG["FD_H"]).

        Truncation error is O(h^2) ~ 1e-8, well below the float32 epsilon (~1e-7),
        so FD_H is near the optimal balance between truncation and rounding error.

        NaN/Inf guards (is_finite) are applied after differentiation to prevent
        degenerate parameter configurations from crashing the training graph. These
        should not trigger in well-behaved runs; persistent NaNs indicate that the
        optimizer has reached a numerically singular configuration.
        """
        r = tf.maximum(tf.cast(r, tf.float32), EPS)
        h = FD_H

        def f(rr):
            return self.beta_r_piecewise(rr, A, B, R0)

        b0 = f(r)
        rp = r + h
        rm = tf.maximum(r - h, EPS)
        bp = f(rp)
        bm = f(rm)
        # First derivative: central difference  b'(r) ≈ (b(r+h) - b(r-h)) / 2h
        b1 = (bp - bm) / (F32(2.0) * h)
        # Second derivative: b''(r) ≈ (b(r+h) - 2*b(r) + b(r-h)) / h^2
        b2 = (bp - F32(2.0) * b0 + bm) / (h * h)
        # Replace any NaN/Inf with zero to keep the graph stable.
        b0 = tf.where(tf.math.is_finite(b0), b0, tf.zeros_like(b0))
        b1 = tf.where(tf.math.is_finite(b1), b1, tf.zeros_like(b1))
        b2 = tf.where(tf.math.is_finite(b2), b2, tf.zeros_like(b2))
        return b0, b1, b2

    @tf.function
    def compute_components_on_coords(self, X, Y, Z, T, A, B, R0):
        """Evaluate the Einstein-tensor-derived stress components on arbitrary coordinates.

        This helper is used both on the training grid and during post-processing on
        diagnostic planes. Keeping the coordinate-based evaluation in one place helps
        ensure that optimization and rerendering use exactly the same tensor algebra.
        """
        X = tf.reshape(tf.cast(X, tf.float32), [-1])
        Y = tf.reshape(tf.cast(Y, tf.float32), [-1])
        Z = tf.reshape(tf.cast(Z, tf.float32), [-1])
        T = tf.reshape(tf.cast(T, tf.float32), [-1])
        r = self.r_geom(X, Y, Z)
        b0, b1, b2 = self.beta_and_derivs(r, A, B, R0)
        v = self.V(T)

        # Pre-compute powers of r and coordinate products used in the Einstein tensor.
        # Small EPS denominators prevent division-by-zero at the origin (r=0).
        r2 = tf.maximum(r * r, EPS)
        r4 = tf.pow(tf.maximum(r, EPS), F32(4.0))
        r5 = tf.pow(tf.maximum(r, EPS), F32(5.0))
        x2 = X * X
        y2 = Y * Y
        z2 = Z * Z
        xy = X * Y
        xz = X * Z
        yz = Y * Z

        # Einstein tensor components G_{mu nu} for the uniformly-translated warp metric.
        #
        # The metric is ds^2 = -dt^2 + (dx - v*beta(r)*dt_z)^2 + dy^2 + dz^2 where
        # beta(r) is the radial shift function and v = V(t) is the bubble speed along z.
        # Here dt_z denotes the z-component of the translation direction.
        #
        # These expressions are the closed-form EFE components derived from that metric
        # ansatz in Cartesian coordinates (r = sqrt(x^2 + y^2 + z^2)).
        # b0 = beta(r),  b1 = d(beta)/dr,  b2 = d^2(beta)/dr^2.
        #
        # The stress-energy tensor is T_{mu nu} = G_{mu nu} / (8*pi) in geometrized units.

        # G_tt: energy density source term.
        Gtt = (b0 * (b0 + F32(2.0) * r * b1)) / r2
        Gxx = (
            (-x2) * r * b0 * b0
            - (y2 + z2) * tf.pow(r, F32(3.0)) * b1 * b1
            - r * r * b0 * (F32(2.0) * r * r * b1 + (y2 + z2) * r * b2)
            + Z
            * v
            * (
                -(F32(4.0) * x2 + y2 + z2) * b0
                + r * (F32(4.0) * x2 + y2 + z2) * b1
                + (y2 + z2) * r * r * b2
            )
        ) / (r5 + EPS)
        Gxy = (
            xy
            * (
                -F32(3.0) * Z * v * b0
                + r * (-b0 * b0 + F32(3.0) * Z * v * b1)
                - Z * r * r * v * b2
                + tf.pow(r, F32(3.0)) * (b1 * b1 + b0 * b2)
            )
        ) / (r5 + EPS)
        Gxz = (
            X
            * v
            * (
                (x2 + y2 - F32(2.0) * z2) * b0
                - r * (x2 + y2 - F32(2.0) * z2) * b1
                - z2 * r * r * b2
            )
        ) / (r5 + EPS) + (xz * (-b0 * b0 + r * r * b1 * b1 + r * r * b0 * b2)) / (r4 + EPS)
        Gyy = (
            (-y2) * r * b0 * b0
            - (x2 + z2) * tf.pow(r, F32(3.0)) * b1 * b1
            - r * r * b0 * (F32(2.0) * r * r * b1 + (x2 + z2) * r * b2)
            + Z
            * v
            * (
                -(x2 + F32(4.0) * y2 + z2) * b0
                + r * (x2 + F32(4.0) * y2 + z2) * b1
                + (x2 + z2) * r * r * b2
            )
        ) / (r5 + EPS)
        Gyz = (
            Y
            * v
            * (
                (x2 + y2 - F32(2.0) * z2) * b0
                - r * (x2 + y2 - F32(2.0) * z2) * b1
                - z2 * r * r * b2
            )
        ) / (r5 + EPS) + (yz * (-b0 * b0 + r * r * b1 * b1 + r * r * b0 * b2)) / (r4 + EPS)
        Gzz = (
            (-z2) * r * b0 * b0
            - (x2 + y2) * tf.pow(r, F32(3.0)) * b1 * b1
            - r * r * b0 * (F32(2.0) * r * r * b1 + (x2 + y2) * r * b2)
            + Z
            * v
            * (
                (x2 + y2 - F32(2.0) * z2) * b0
                - r * (x2 + y2 - F32(2.0) * z2) * b1
                + (x2 + y2) * r * r * b2
            )
        ) / (r5 + EPS)

        # Convert G_{mu nu} to stress-energy T_{mu nu} = G_{mu nu} / (8*pi).
        factor = F32(1.0) / (F32(8.0) * PI32)
        rho = Gtt * factor   # energy density
        Px = Gxx * factor    # x-pressure
        Py = Gyy * factor    # y-pressure
        Pz = Gzz * factor    # z-pressure
        Txy = Gxy * factor   # xy shear
        Txz = Gxz * factor   # xz shear
        Tyz = Gyz * factor   # yz shear
        return rho, Px, Py, Pz, Txy, Txz, Tyz, r

    @tf.function
    def compute_components(self, A, B, R0):
        return self.compute_components_on_coords(self.X, self.Y, self.Z, self.T, A, B, R0)

    @tf.function
    def assemble_P_eigs(self, Px, Py, Pz, Txy, Txz, Tyz):
        M = tf.stack(
            [
                tf.stack([Px, Txy, Txz], axis=-1),
                tf.stack([Txy, Py, Tyz], axis=-1),
                tf.stack([Txz, Tyz, Pz], axis=-1),
            ],
            axis=-2,
        )
        e = tf.linalg.eigvalsh(M)
        return e[:, 0], e[:, 1], e[:, 2]

    # Centralize principal-stress margins so loss, diagnostics, and exports agree.
    @tf.function
    def stress_margins_eig(self, rho, Px, Py, Pz, Txy, Txz, Tyz):
        lam1, lam2, lam3 = self.assemble_P_eigs(Px, Py, Pz, Txy, Txz, Tyz)
        lam_min = tf.minimum(lam1, tf.minimum(lam2, lam3))
        zero = tf.zeros_like(lam_min)
        nec_margin = rho + lam_min
        wec_margin = rho + tf.minimum(lam_min, zero)
        dec_margin = rho - tf.maximum(tf.abs(lam1), tf.maximum(tf.abs(lam2), tf.abs(lam3)))
        sec_margin = rho + lam1 + lam2 + lam3
        return nec_margin, wec_margin, dec_margin, sec_margin

    def soft_mask(self, r, A, B, R0, alpha):
        """Return the differentiable shell mask used to weight losses during training.

        The soft mask is a numerical device only: it localizes the objective to the
        shell region while remaining differentiable with respect to the trainable
        parameters. The corresponding hard mask is used later for referee-facing
        success fractions.
        """
        if self.domain_type == 1:
            thr = F32(2.0) / B
            wl = tf.sigmoid(alpha * (r - thr))
            wr = tf.sigmoid(alpha * (self.r_cap - r))
            return wl * wr
        wl = tf.sigmoid(alpha * (r - R0))
        wr = tf.sigmoid(alpha * ((F32(2.0) / B) - r))
        return wl * wr

    def hard_masks(self, r, A, B, R0):
        """Return the exact in-domain region together with its left/right complement.

        The hard mask defines the domain on which exported success fractions are
        reported. It is intentionally separate from the soft mask so that the paper
        can distinguish optimization weights from binary pass/fail diagnostics.
        """
        if self.domain_type == 1:
            thr = F32(2.0) / B
            hard = tf.logical_and(r >= thr, r <= self.r_cap)
            left = r < thr
            right = r > self.r_cap
        else:
            hard = tf.logical_and(r >= R0, r <= (F32(2.0) / B))
            left = r < R0
            right = r > (F32(2.0) / B)
        return hard, left, right

    def physics_loss_all_observers(self, rho, Px, Py, Pz, Txy, Txz, Tyz, w, use_dec=True, dec_w=0.25):
        """Compute the soft physics penalty from principal-stress energy-condition margins.

        The loss does not enforce the energy conditions analytically. Instead it
        penalizes negative margins for NEC/WEC and, optionally, DEC inside the soft
        shell mask. This is the key reason the workflow should be described as a
        physics-penalized optimization rather than as an exact constraint solver.
        """
        nec_margin, wec_margin, dec_margin, _ = self.stress_margins_eig(
            rho, Px, Py, Pz, Txy, Txz, Tyz
        )

        relu = tf.nn.relu

        def penal(margin):
            v = relu(-(margin - (-HARD_TOL)))
            return tf.reduce_mean(w * (v + F32(0.5) * v * v))

        L = penal(nec_margin) + penal(wec_margin)
        if use_dec:
            L += F32(dec_w) * penal(dec_margin)
        return self.physics_scale * L

    @tf.function
    def energy_margins_all_observers(self, Px, Py, Pz, Txy, Txz, Tyz, rho):
        return self.stress_margins_eig(rho, Px, Py, Pz, Txy, Txz, Tyz)

    def success_rates_eig(self, rho, Px, Py, Pz, Txy, Txz, Tyz, r, A, B, R0):
        hard, _, _ = self.hard_masks(r, A, B, R0)
        nec_margin, wec_margin, dec_margin, sec_margin = self.stress_margins_eig(
            rho, Px, Py, Pz, Txy, Txz, Tyz
        )

        def rate(margin):
            masked = tf.boolean_mask(margin, hard)
            if tf.size(masked) == 0:
                return F32(np.nan)
            return F32(100.0) * tf.reduce_mean(tf.cast(masked >= -HARD_TOL, tf.float32))

        # These fractions now match the principal-stress margins exported by the paper pipeline.
        return {
            "nec": rate(nec_margin),
            "wec": rate(wec_margin),
            "dec": rate(dec_margin),
            "sec": rate(sec_margin),
        }

    def smoothing_error_continuous(self, r, A, B, R0, alpha, weights=(1.0, 0.5, 0.5)):
        """Measure how closely the soft mask matches the intended hard shell domain.

        `FN` captures shell points that are insufficiently activated, while `FP_L`
        and `FP_R` capture mask leakage on either side of the intended interface.
        This term regularizes the geometry of the support region rather than the
        stress tensor itself.
        """
        g = self.soft_mask(r, A, B, R0, alpha)
        hard, left, right = self.hard_masks(r, A, B, R0)
        hard_f = tf.cast(hard, tf.float32)
        left_f = tf.cast(left, tf.float32)
        right_f = tf.cast(right, tf.float32)
        n_active = tf.reduce_sum(hard_f) + EPS
        FN = tf.reduce_sum(hard_f * (F32(1.0) - g)) / n_active
        FP_L = tf.reduce_sum(left_f * g) / n_active
        FP_R = tf.reduce_sum(right_f * g) / n_active
        w0, w1, w2 = [F32(w) for w in weights]
        return FN, FP_L, FP_R, w0 * FN + w1 * FP_L + w2 * FP_R

    def train(self, num_epochs=300, lr=3e-3, print_interval=20):
        """Run the main optimization stage starting from the current parameter snapshot.

        The history exported here is the authoritative training record used by the
        paper figures. In particular, the success fractions stored in `hist` are the
        principal-stress diagnostics on the hard mask, while the loss terms come from
        the soft-mask objective that actually drives the optimizer.
        """
        vars_ = [self.A_raw, self.B_raw]
        if self.R0_raw is not None:
            vars_.append(self.R0_raw)
        if self.alpha_raw is not None:
            vars_.append(self.alpha_raw)

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        hist = {
            "loss_total": [],
            "loss_phys": [],
            "loss_smooth": [],
            "loss_invalpha": [],
            "loss_shear": [],
            "alpha": [],
            "A": [],
            "B": [],
            "R0": [],
            "E_FN%": [],
            "E_FP_L%": [],
            "E_FP_R%": [],
            "E_tot%": [],
            "succ_nec": [],
            "succ_wec": [],
            "succ_dec": [],
            "succ_sec": [],
            "nec_margin_soft": [],
            "wec_margin_soft": [],
            "dec_margin_soft": [],
            "sec_margin_soft": [],
        }
        log("\nTraining (principal-stress diagnostics)")
        t0 = time.time()
        pbar = tqdm_opt(range(1, num_epochs + 1), desc="Main Training", unit="epoch")
        for ep in pbar:
            with tf.GradientTape() as tape:
                A, B, R0 = self.get_ABR()
                alpha = self.alpha_current(ep, num_epochs)
                rho, Px, Py, Pz, Txy, Txz, Tyz, r = self.compute_components(A, B, R0)
                w = self.soft_mask(r, A, B, R0, alpha)

                # The loss operates on principal-stress margins, not on component-wise Type-I checks.
                L_phys = self.physics_loss_all_observers(
                    rho,
                    Px,
                    Py,
                    Pz,
                    Txy,
                    Txz,
                    Tyz,
                    w,
                    use_dec=self.use_dec_loss,
                    dec_w=self.dec_loss_weight,
                )

                if self.L_shear > 0.0:
                    shear_pen = self.L_shear * tf.reduce_mean(w * (Txy * Txy + Txz * Txz + Tyz * Tyz))
                else:
                    shear_pen = F32(0.0)

                FN, FP_L, FP_R, E_tot = self.smoothing_error_continuous(r, A, B, R0, alpha)
                breach = tf.nn.relu(E_tot - F32(self.smoothing_target))
                L_smooth = F32(self.L_breach) * (breach**2)
                L_alpha = F32(self.L_inv_alpha) / (alpha + F32(1e-6))
                L_total = L_phys + L_smooth + L_alpha + shear_pen

            grads = tape.gradient(L_total, vars_)
            grads = [tf.clip_by_value(g, -0.5, 0.5) if g is not None else g for g in grads]
            opt.apply_gradients(zip(grads, vars_))

            succ = self.success_rates_eig(rho, Px, Py, Pz, Txy, Txz, Tyz, r, A, B, R0)
            loss_val = float(L_total.numpy())
            phys_val = float(L_phys.numpy())
            hist["loss_total"].append(loss_val)
            hist["loss_phys"].append(phys_val)
            hist["loss_smooth"].append(float(L_smooth.numpy()))
            hist["loss_invalpha"].append(float(L_alpha.numpy()))
            hist["loss_shear"].append(float(shear_pen.numpy()))
            hist["alpha"].append(float(alpha.numpy()))
            hist["A"].append(float(A.numpy()))
            hist["B"].append(float(B.numpy()))
            hist["R0"].append(float(R0.numpy()) if R0 is not None else np.nan)
            hist["E_FN%"].append(float(FN.numpy() * 100.0))
            hist["E_FP_L%"].append(float(FP_L.numpy() * 100.0))
            hist["E_FP_R%"].append(float(FP_R.numpy() * 100.0))
            hist["E_tot%"].append(float(E_tot.numpy() * 100.0))
            for key in ("nec", "wec", "dec", "sec"):
                value = succ[key].numpy()
                hist[f"succ_{key}"].append(float(value if np.isfinite(value) else np.nan))

            nec_margin, wec_margin, dec_margin, sec_margin = self.energy_margins_all_observers(
                Px, Py, Pz, Txy, Txz, Tyz, rho
            )
            hist["nec_margin_soft"].append(float(tf.reduce_mean(w * nec_margin).numpy()))
            hist["wec_margin_soft"].append(float(tf.reduce_mean(w * wec_margin).numpy()))
            hist["dec_margin_soft"].append(float(tf.reduce_mean(w * dec_margin).numpy()))
            hist["sec_margin_soft"].append(float(tf.reduce_mean(w * sec_margin).numpy()))

            avg_s = np.nanmean(
                [
                    hist["succ_nec"][-1],
                    hist["succ_wec"][-1],
                    hist["succ_dec"][-1],
                    hist["succ_sec"][-1],
                ]
            )
            if USE_TQDM:
                pbar.set_postfix(
                    {"loss": f"{loss_val:.3e}", "phys": f"{phys_val:.3e}", "a": f"{hist['alpha'][-1]:.1f}", "succ": f"{avg_s:.1f}%"}
                )
            if ((print_interval and ep % print_interval == 0) or ep == 1) and (not QUIET):
                if self.domain_type == 1:
                    thr_now = float(2.0 / float(B.numpy()))
                    msg_dom = f"band: [{thr_now:.4f}, {float(self.r_cap.numpy()):.4f}]"
                else:
                    msg_dom = f"R0={hist['R0'][-1]:.3f}, 2/B={(2.0 / float(B.numpy())):.3f}"
                log(
                    f"Ep {ep:4d}/{num_epochs} | Loss:{loss_val:.3e} | Phys:{phys_val:.3e} | "
                    f"A={hist['A'][-1]:.3f} B={hist['B'][-1]:.3f} | {msg_dom} | "
                    f"a={hist['alpha'][-1]:.1f} | E_tot%={hist['E_tot%'][-1]:.2f} | "
                    f"Succ(NEC/WEC/DEC/SEC)={avg_s:.1f}%"
                )
        if USE_TQDM:
            pbar.close()
        log(f"\nTraining time: {time.time() - t0:.1f}s")
        return hist

    def train_quick(self, num_epochs=60, lr=3e-3):
        """Run a short optimization used only during Monte-Carlo pretraining.

        The goal is not to produce publishable curves, but to rank candidate seeds so
        that the main run starts from a lower-loss basin. This is why some parameter
        plots in the paper can appear nearly stationary from epoch one: the visible
        run begins after this pretraining stage has already selected a good snapshot.
        """
        vars_ = [self.A_raw, self.B_raw]
        if self.R0_raw is not None:
            vars_.append(self.R0_raw)
        if self.alpha_raw is not None:
            vars_.append(self.alpha_raw)

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        total = None
        pbar = tqdm_opt(range(1, num_epochs + 1), desc="Pre-train Mini-Epochs", unit="epoch")
        for ep in pbar:
            with tf.GradientTape() as tape:
                A, B, R0 = self.get_ABR()
                alpha = self.alpha_current(ep, num_epochs)
                rho, Px, Py, Pz, Txy, Txz, Tyz, r = self.compute_components(A, B, R0)
                w = self.soft_mask(r, A, B, R0, alpha)
                L_phys = self.physics_loss_all_observers(
                    rho, Px, Py, Pz, Txy, Txz, Tyz, w, use_dec=True, dec_w=0.25
                )
                if self.L_shear > 0.0:
                    shear_pen = self.L_shear * tf.reduce_mean(w * (Txy * Txy + Txz * Txz + Tyz * Tyz))
                else:
                    shear_pen = F32(0.0)
                FN, FP_L, FP_R, E_tot = self.smoothing_error_continuous(r, A, B, R0, alpha)
                breach = tf.nn.relu(E_tot - F32(self.smoothing_target))
                L_smooth = F32(self.L_breach) * (breach**2)
                L_alpha = F32(self.L_inv_alpha) / (alpha + F32(1e-6))
                total = L_phys + L_smooth + L_alpha + shear_pen
            grads = tape.gradient(total, vars_)
            grads = [tf.clip_by_value(g, -0.5, 0.5) if g is not None else g for g in grads]
            opt.apply_gradients(zip(grads, vars_))
            if USE_TQDM:
                pbar.set_postfix({"loss": f"{float(total.numpy()):.3e}"})
        if USE_TQDM:
            pbar.close()
        return float(total.numpy())

    def pretrain_random_inits(self, trials, pre_epochs, lr, ranges):
        """Search over random initial seeds before the main optimization begins.

        This stage is deliberately simple: sample `(A, B, R0)` from broad ranges, run
        a short quick-train, and keep the best snapshot. The chosen seed is recorded
        only implicitly through the final run history, so the manuscript should refer
        to this stage whenever the main training curves start already near a plateau.
        """
        if trials <= 0:
            return
        log(f"\nMonte-Carlo pre-training: {trials} trials x {pre_epochs} mini-epochs")
        best = math.inf
        snap = None
        B_min_val = 2.0 / (float(self.r_cap.numpy()) - float(RCAP_EPS.numpy())) if self.domain_type == 1 else None
        pbar_trials = tqdm_opt(range(1, trials + 1), desc="Pre-train Trials", unit="trial")
        for i in pbar_trials:
            A0 = float(np.random.uniform(*ranges["A"]))
            B0 = float(np.random.uniform(*ranges["B"]))
            if self.domain_type == 1 and B_min_val is not None:
                B0 = max(B0, B_min_val + 1e-5)
            if self.domain_type == 2:
                upper = max(2.0 / B0 - 2.0 * CONFIG["R0_MARGIN"], CONFIG["R0_MARGIN"] + 1e-3)
                R0samp = float(np.random.uniform(*ranges["R0"]))
                R00 = float(np.clip(R0samp, CONFIG["R0_MARGIN"] + 1e-4, upper - 1e-4))
            else:
                R00 = None
            self.set_from_values(A0, B0, R00)
            if self.alpha_raw is not None:
                self.alpha_raw.assign(inv_softplus_pos(max(self.alpha_init - self.alpha_floor_start, 1e-3)))
            loss_end = self.train_quick(num_epochs=pre_epochs, lr=lr)
            if loss_end < best:
                best = loss_end
                snap = {
                    "A_raw": float(self.A_raw.numpy()),
                    "B_raw": float(self.B_raw.numpy()),
                    "R0_raw": float(self.R0_raw.numpy()) if self.R0_raw is not None else None,
                    "alpha_raw": float(self.alpha_raw.numpy()) if self.alpha_raw is not None else None,
                }
            if USE_TQDM:
                pbar_trials.set_postfix({"loss_end": f"{loss_end:.3e}", "best": f"{best:.3e}"})
            log(f" trial {i:02d}/{trials} | loss_final={loss_end:.3e} | best={best:.3e}")
        if USE_TQDM:
            pbar_trials.close()
        if snap is not None:
            self.A_raw.assign(snap["A_raw"])
            self.B_raw.assign(snap["B_raw"])
            if self.R0_raw is not None:
                self.R0_raw.assign(snap["R0_raw"])
            if self.alpha_raw is not None:
                self.alpha_raw.assign(snap["alpha_raw"])
        log("Best MC snapshot restored.\n")

    def final_report(self, hist):
        """Summarize the final optimized snapshot in referee-facing diagnostic terms.

        The report intentionally uses the same principal-stress quantities exported to
        CSV so that textual summaries, tabulated data, and plots all describe the same
        physical diagnostics.
        """
        A = F32(hist["A"][-1])
        B = F32(hist["B"][-1])
        alpha_final = F32(hist["alpha"][-1])
        R0 = F32(hist["R0"][-1]) if self.domain_type == 2 else None

        rho, Px, Py, Pz, Txy, Txz, Tyz, r = self.compute_components(
            A, B, R0 if R0 is not None else F32(1.0)
        )
        hard, _, _ = self.hard_masks(r, A, B, R0 if R0 is not None else F32(1.0))
        n_active = int(tf.reduce_sum(tf.cast(hard, tf.int32)).numpy())

        succ_eig = self.success_rates_eig(
            rho, Px, Py, Pz, Txy, Txz, Tyz, r, A, B, R0 if R0 is not None else F32(1.0)
        )
        nec_margin, wec_margin, dec_margin, sec_margin = self.energy_margins_all_observers(
            Px, Py, Pz, Txy, Txz, Tyz, rho
        )

        lines = []
        lines.append("\n" + "=" * 65)
        lines.append("FINAL REPORT - Principal-stress diagnostics")
        lines.append("=" * 65)
        if self.domain_type == 1:
            thr_val = float((F32(2.0) / B).numpy())
            lines.append(
                f"A={float(A.numpy()):.6f} | B={float(B.numpy()):.6f} | alpha_final={float(alpha_final.numpy()):.2f}"
            )
            lines.append(f" Hard band (capped to box): [{thr_val:.4f}, {float(self.r_cap.numpy()):.4f}]")
        else:
            lines.append(
                f"A={float(A.numpy()):.6f} | B={float(B.numpy()):.6f} | "
                f"R0={float(R0.numpy()):.6f} | alpha_final={float(alpha_final.numpy()):.2f}"
            )
            lines.append(
                f" Domain duro: [{float(R0.numpy()):.4f}, {float((F32(2.0) / B).numpy()):.4f}] (capped to box)"
            )
        lines.append(f"Hard-domain points: {n_active}\n")
        lines.append("Success fractions (HARD mask):")
        for key, label in (("nec", "NEC"), ("wec", "WEC"), ("dec", "DEC"), ("sec", "SEC")):
            lines.append(f" - {label}: {float(succ_eig[key].numpy()):.2f}%")

        lines.append("\nPrincipal-stress margins (unweighted snapshot):")
        lines.append(f" - <NEC_margin> ~ {float(tf.reduce_mean(nec_margin).numpy()):.6e}")
        lines.append(f" - <WEC_margin> ~ {float(tf.reduce_mean(wec_margin).numpy()):.6e}")
        lines.append(f" - <DEC_margin> ~ {float(tf.reduce_mean(dec_margin).numpy()):.6e}")
        lines.append(f" - <SEC_margin> ~ {float(tf.reduce_mean(sec_margin).numpy()):.6e}")

        report_text = "\n".join(lines)
        log(report_text)
        self._final_succ_eig = [float(succ_eig[key].numpy()) for key in ("nec", "wec", "dec", "sec")]

        final_params = {
            "A": float(A.numpy()),
            "B": float(B.numpy()),
            "R0": float(R0.numpy()) if self.domain_type == 2 else None,
            "alpha": float(alpha_final.numpy()),
        }
        return report_text, final_params

    @staticmethod
    def _show():
        try:
            plt.tight_layout()
            plt.show()
        except Exception:
            pass
        finally:
            plt.close("all")

    def plot_all(self, h):
        if not h["loss_total"]:
            return
        E = range(1, len(h["loss_total"]) + 1)
        plt.figure(figsize=(11, 4))
        plt.semilogy(E, h["loss_total"], label="Total")
        plt.semilogy(E, h["loss_phys"], label="Physics (scaled)")
        plt.semilogy(E, h["loss_smooth"], label="Smooth (breach^2)")
        plt.semilogy(E, h["loss_invalpha"], label="1/alpha")
        if any(np.array(h["loss_shear"]) > 0):
            plt.semilogy(E, h["loss_shear"], label="Shear")
        plt.title("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        self._show()

        plt.figure(figsize=(11, 4))
        for key in ("succ_nec", "succ_wec", "succ_dec", "succ_sec"):
            plt.plot(E, h[key], label=key.replace("succ_", "").upper())
        plt.ylim(0, 100)
        plt.title("Success in HARD domain (%)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        self._show()

        plt.figure(figsize=(11, 4))
        plt.plot(E, h["alpha"], label="alpha")
        plt.twinx()
        plt.plot(E, h["A"], label="A")
        plt.plot(E, h["B"], label="B")
        if not np.isnan(h["R0"][-1]):
            plt.plot(E, h["R0"], label="R0")
        plt.title("Alpha & Params")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        self._show()

        plt.figure(figsize=(11, 4))
        plt.plot(E, h["E_tot%"], label="E_tot%")
        plt.plot(E, h["E_FN%"], label="FN%")
        plt.plot(E, h["E_FP_L%"], label="FP_L%")
        plt.plot(E, h["E_FP_R%"], label="FP_R%")
        plt.title("Smoothing components (%)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        self._show()

        labels = ["NEC", "WEC", "DEC", "SEC"]
        finals_h = getattr(
            self,
            "_final_succ_eig",
            [h["succ_nec"][-1], h["succ_wec"][-1], h["succ_dec"][-1], h["succ_sec"][-1]],
        )
        plt.figure(figsize=(8, 4))
        bars = plt.bar(labels, finals_h, edgecolor="black")
        for bar, val in zip(bars, finals_h):
            if np.isfinite(val):
                plt.text(bar.get_x() + bar.get_width() / 2, max(val, 0) + 1, f"{val:.1f}%", ha="center", va="bottom")
        plt.ylim(0, 100)
        plt.title("Final Success (HARD, principal-stress)")
        plt.grid(True, axis="y", alpha=0.3)
        self._show()

    def export_history_csvs(self, history, base_name):
        """Write the training history in the normalized schema used across the repo.

        These CSVs are the data contract shared by verification, plotting, and paper
        audits. Any future change to column meanings should therefore be mirrored in
        the manuscript and in the downstream analysis scripts.
        """
        with open(f"{base_name}_losses.csv", "w", encoding="utf-8") as f:
            f.write("epoch,total_loss,physics_loss,reg_breach,inv_alpha,E_soft_%,shear_pen\n")
            for i in range(len(history["loss_total"])):
                f.write(
                    f"{i+1},{history['loss_total'][i]},{history['loss_phys'][i]},"
                    f"{history['loss_smooth'][i]},{history['loss_invalpha'][i]},"
                    f"{history['E_tot%'][i]},{history['loss_shear'][i]}\n"
                )
        with open(f"{base_name}_parameters.csv", "w", encoding="utf-8") as f:
            has_r0 = not np.isnan(history["R0"][-1])
            header = "epoch,A,B," + ("R0," if has_r0 else "") + "alpha\n"
            f.write(header)
            for i in range(len(history["A"])):
                row = f"{i+1},{history['A'][i]},{history['B'][i]},"
                if has_r0:
                    row += f"{history['R0'][i]},"
                row += f"{history['alpha'][i]}\n"
                f.write(row)
        with open(f"{base_name}_success_rates.csv", "w", encoding="utf-8") as f:
            # Export names now match the eigenvalue-based principal-stress diagnostics.
            f.write("epoch,nec,wec,dec,sec,nec_margin_soft,wec_margin_soft,dec_margin_soft,sec_margin_soft\n")
            L = len(history["loss_total"])
            for i in range(L):
                f.write(
                    ",".join(
                        [
                            str(i + 1),
                            f"{history['succ_nec'][i]}",
                            f"{history['succ_wec'][i]}",
                            f"{history['succ_dec'][i]}",
                            f"{history['succ_sec'][i]}",
                            f"{history['nec_margin_soft'][i]}",
                            f"{history['wec_margin_soft'][i]}",
                            f"{history['dec_margin_soft'][i]}",
                            f"{history['sec_margin_soft'][i]}",
                        ]
                    )
                    + "\n"
                )


def build_run_metadata(cfg, final_params, base_name):
    """Store the run configuration needed to reproduce diagnostics without guessing.

    The metadata is the referee-facing contract for this repository: post-processing,
    verification, and external re-runs should be able to reconstruct the same
    diagnostics without reaching back into CONFIG defaults or notebook state.
    """
    c0, c1 = cfg.get("V_COEFFS", (0.0, 0.1))
    snapshot_velocity = current_snapshot_velocity(cfg)
    return {
        "domain_type": int(cfg["DOMAIN_TYPE"]),
        "v_mode": str(cfg.get("V_MODE", "linear")),
        "velocity": snapshot_velocity,
        "time": float(cfg["T_RANGE"][0]),
        "t_range": [float(cfg["T_RANGE"][0]), float(cfg["T_RANGE"][1])],
        "seed": int(cfg["SEED"]),
        "N_xyz": int(cfg["N_XYZ"]),
        "N_t": int(cfg.get("N_T", 1)),
        "xyz_range": [float(cfg["XYZ_RANGE"][0]), float(cfg["XYZ_RANGE"][1])],
        "final_parameters": final_params,
        # Velocity settings are duplicated here because the compact basename only keeps
        # the snapshot tag, while reproducing a linear run requires the underlying law.
        "velocity_settings": {
            "mode": str(cfg.get("V_MODE", "linear")),
            "constant_velocity": float(cfg["VELOCITY"]),
            "v_coeffs": [float(c0), float(c1)],
            "snapshot_velocity": snapshot_velocity,
        },
        "alpha_settings": {
            "mode": str(cfg["ALPHA_MODE"]),
            "init": float(cfg["ALPHA_INIT"]),
            "floor_start": float(cfg["ALPHA_FLOOR_START"]),
            "floor_end": float(cfg["ALPHA_FLOOR_END"]),
            "warmup_frac": float(cfg["ALPHA_WARMUP_FRAC"]),
        },
        # These weights affect what the optimizer is actually minimizing, so they must
        # travel with the bundle if we want the run to remain scientifically auditable.
        "physics_settings": {
            "physics_scale": float(cfg["PHYSICS_SCALE"]),
            "L_breach": float(cfg["L_BREACH"]),
            "L_inv_alpha": float(cfg["L_INV_ALPHA"]),
            "L_shear": float(cfg["L_SHEAR"]),
            "use_dec_loss": bool(cfg.get("USE_DEC_LOSS", True)),
            "dec_loss_weight": float(cfg.get("DEC_LOSS_WEIGHT", 0.25)),
            "smoothing_target_pct": float(cfg["SMOOTHING_TARGET_PCT"]),
            "hard_tol": float(cfg["HARD_TOL"]),
        },
        # Optimizer settings record the numerical recipe used to produce the bundle.
        "optimizer_settings": {
            "num_epochs": int(cfg["NUM_EPOCHS"]),
            "lr": float(cfg["LR"]),
            "pretrain_trials": int(cfg["PRETRAIN_TRIALS"]),
            "pretrain_epochs": int(cfg["PRETRAIN_EPOCHS"]),
            "pretrain_lr": float(cfg["PRETRAIN_LR"]),
            "overwrite_outputs": bool(cfg.get("OVERWRITE_OUTPUTS", False)),
        },
        "file_basename": base_name,
    }

def run_cpu(cfg=CONFIG):
    """Run one optimizer snapshot and export an auditable bundle.

    This is the main entry point used by the CLI and by generate_run_bundle.py. The
    function deep-copies the input configuration so callers can safely override
    values without mutating the module-level defaults seen by later runs.
    """
    cfg = copy.deepcopy(cfg)
    base = build_base_name(cfg)
    ensure_output_paths_available(base, overwrite=bool(cfg.get("OVERWRITE_OUTPUTS", False)))

    tr = EinsteinTrainerCPU(
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

    tr.create_variables(A_init=cfg["A_INIT"], B_init=cfg["B_INIT"], R0_init=cfg["R0_INIT"])

    if cfg["PRETRAIN_TRIALS"] > 0:
        tr.pretrain_random_inits(cfg["PRETRAIN_TRIALS"], cfg["PRETRAIN_EPOCHS"], cfg["PRETRAIN_LR"], cfg["INIT_RANGES"])
    hist = tr.train(num_epochs=cfg["NUM_EPOCHS"], lr=cfg["LR"], print_interval=cfg["PRINT_EVERY"])

    report_text, final_params = tr.final_report(hist)
    if cfg.get("SHOW_PLOTS", True):
        tr.plot_all(hist)

    tr.export_history_csvs(hist, base)

    with open(f"{base}_final_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(f"{base}_final_params.csv", "w", encoding="utf-8") as f:
        f.write("A,B,R0,alpha\n")
        r0_out = "" if final_params["R0"] is None else final_params["R0"]
        f.write(f"{final_params['A']},{final_params['B']},{r0_out},{final_params['alpha']}\n")

    metadata = build_run_metadata(cfg, final_params, base)
    with open(f"{base}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    log(f"\nCSV export base: {base}_*.csv")
    log(f"Final report saved to: {base}_final_report.txt")
    log(f"Final params saved to: {base}_final_params.csv")
    log(f"Metadata saved to: {base}_metadata.json")

    return tr, hist

def _cli():
    p = argparse.ArgumentParser(description="Run Einstein optimizer for a single (domain, v, t) snapshot.")
    p.add_argument("--domain", type=int, choices=[1, 2], default=CONFIG["DOMAIN_TYPE"], help="1: single shell, 2: double shell")
    p.add_argument("--v", type=float, default=CONFIG["VELOCITY"], help="bubble speed along +z")
    p.add_argument("--t", type=float, default=CONFIG["T_RANGE"][0], help="time snapshot; single frame")
    p.add_argument("--seed", type=int, default=None, help="override RNG seed")
    p.add_argument("--overwrite", action="store_true", help="allow replacing an existing bundle with the same basename")
    args = p.parse_args()

    cfg = copy.deepcopy(CONFIG)
    cfg["DOMAIN_TYPE"] = int(args.domain)
    cfg["VELOCITY"] = float(args.v)
    cfg["T_RANGE"] = (float(args.t), float(args.t))
    cfg["N_T"] = 1
    if args.seed is not None:
        cfg["SEED"] = int(args.seed)
    cfg["OVERWRITE_OUTPUTS"] = bool(args.overwrite)
    tf.random.set_seed(cfg["SEED"])
    np.random.seed(cfg["SEED"])
    run_cpu(cfg)


if __name__ == "__main__":
    _cli()


