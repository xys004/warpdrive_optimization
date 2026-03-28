"""Shared principal-stress eigenvalue helpers for the warp-bubble pipeline.

These utilities are imported by both the optimizer (einstein_optimizer.py) and
the post-processing plotter (postprocess_plots.py) to guarantee that energy-
condition margins are computed identically in both stages.

Physical context
----------------
For a stress-energy tensor T^{mu nu} associated with an Alcubierre-type warp
metric, the spatial pressure tensor P_{ij} is symmetric. Its three real
eigenvalues (principal pressures lambda_1 <= lambda_2 <= lambda_3) determine
whether the matter content violates standard energy conditions:

- NEC (Null Energy Condition):    rho + lambda_min >= 0
- WEC (Weak Energy Condition):    rho >= 0  AND  rho + lambda_i >= 0  for all i
                                   => equivalent to NEC when rho >= 0,
                                      otherwise tighter: rho + min(lambda, 0) >= 0
- DEC (Dominant Energy Condition): rho - |lambda|_max >= 0
- SEC (Strong Energy Condition):   rho + sum(lambda_i) >= 0

A positive margin means the condition is satisfied at that point.
"""

import tensorflow as tf


def assemble_pressure_eigenvalues(Px, Py, Pz, Txy, Txz, Tyz):
    """Compute the three principal pressures of the spatial stress tensor.

    Constructs the symmetric 3x3 pressure matrix from its six independent
    components and returns its real eigenvalues in ascending order via
    tf.linalg.eigvalsh (guaranteed real for symmetric inputs).

    Parameters
    ----------
    Px, Py, Pz : tf.Tensor, shape (N,)
        Diagonal pressure components in x, y, z directions.
    Txy, Txz, Tyz : tf.Tensor, shape (N,)
        Off-diagonal (shear) components of the pressure tensor.

    Returns
    -------
    lam1, lam2, lam3 : tf.Tensor, shape (N,)
        Principal pressures sorted in ascending order (lam1 <= lam2 <= lam3).
    """
    # Build the symmetric 3x3 pressure matrix for each grid point.
    matrix = tf.stack(
        [
            tf.stack([Px, Txy, Txz], axis=-1),
            tf.stack([Txy, Py, Tyz], axis=-1),
            tf.stack([Txz, Tyz, Pz], axis=-1),
        ],
        axis=-2,
    )
    # eigvalsh returns eigenvalues in ascending order for symmetric matrices.
    eigenvalues = tf.linalg.eigvalsh(matrix)
    return eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]


def principal_stress_margins(rho, Px, Py, Pz, Txy, Txz, Tyz):
    """Compute the four standard energy-condition margins at every grid point.

    All margins are signed: positive => condition satisfied, negative => violated.
    These are the quantities exported to CSV and used in the paper's success-
    fraction diagnostics.

    Parameters
    ----------
    rho : tf.Tensor, shape (N,)
        Energy density (T^{tt} / 8pi in geometrized units).
    Px, Py, Pz : tf.Tensor, shape (N,)
        Diagonal pressure components.
    Txy, Txz, Tyz : tf.Tensor, shape (N,)
        Off-diagonal shear components.

    Returns
    -------
    nec_margin : tf.Tensor
        NEC margin: rho + lambda_min.
    wec_margin : tf.Tensor
        WEC margin: rho + min(lambda_min, 0).  Equal to NEC when rho >= 0.
    dec_margin : tf.Tensor
        DEC margin: rho - |lambda|_max.
    sec_margin : tf.Tensor
        SEC margin: rho + lambda_1 + lambda_2 + lambda_3.
    """
    lam1, lam2, lam3 = assemble_pressure_eigenvalues(Px, Py, Pz, Txy, Txz, Tyz)
    lam_min = tf.minimum(lam1, tf.minimum(lam2, lam3))
    zero = tf.zeros_like(lam_min)
    nec_margin = rho + lam_min
    # WEC requires rho >= 0 in addition to NEC; the extra term captures that.
    wec_margin = rho + tf.minimum(lam_min, zero)
    dec_margin = rho - tf.maximum(tf.abs(lam1), tf.maximum(tf.abs(lam2), tf.abs(lam3)))
    # SEC involves the trace of the pressure tensor (sum of all eigenvalues).
    sec_margin = rho + lam1 + lam2 + lam3
    return nec_margin, wec_margin, dec_margin, sec_margin


def hawking_ellis_type1_diagnostic(rho, Px, Py, Pz, Txy, Txz, Tyz,
                                   qx=None, qy=None, qz=None):
    """Verify Hawking-Ellis Type I classification at every grid point.

    For a stress-energy tensor T^a_b in an orthonormal frame:
      T^a_b = | -rho   q_j  |
              | -q_i   P_ij |

    Type I requires:
    1. q_i = 0 (block-diagonal, no energy flux)
    2. P_ij real-symmetric (guaranteed) with real eigenvalues
    3. Timelike eigenvalue (-rho) distinct from spatial eigenvalues lambda_i

    When Type I holds, the pointwise energy conditions for ALL observers
    reduce to the algebraic principal-stress conditions:
      WEC: rho >= 0 AND rho + lambda_i >= 0
      NEC: rho + lambda_i >= 0
      DEC: rho >= |lambda_i|
      SEC: rho + sum(lambda_i) >= 0 AND rho + lambda_i >= 0

    Parameters
    ----------
    rho : tf.Tensor, shape (N,)
        Energy density.
    Px, Py, Pz, Txy, Txz, Tyz : tf.Tensor, shape (N,)
        Spatial pressure components.
    qx, qy, qz : tf.Tensor or None
        Energy flux components. If None, assumed zero (as proved analytically
        for the uniform-translation zero-vorticity class).

    Returns
    -------
    is_type1 : tf.Tensor, shape (N,), bool
        True where the tensor is Type I within tolerance.
    flux_norm : tf.Tensor, shape (N,)
        ||q||, the norm of the energy flux vector. Should be ~0 for Type I.
    eigenvalue_gap : tf.Tensor, shape (N,)
        min_i |rho + lambda_i|, the gap between timelike and spatial eigenvalues.
        Positive means non-degenerate Type I.
    type1_fraction : tf.Tensor, scalar
        Fraction of grid points classified as Type I.
    """
    # Energy flux norm
    if qx is None:
        flux_norm = tf.zeros_like(rho)
    else:
        flux_norm = tf.sqrt(qx * qx + qy * qy + qz * qz + 1e-30)

    # Spatial eigenvalues
    lam1, lam2, lam3 = assemble_pressure_eigenvalues(Px, Py, Pz, Txy, Txz, Tyz)

    # The timelike eigenvalue is -rho. Type I requires it to be distinct from
    # spatial eigenvalues.  The spatial eigenvalues are lambda_i.
    # We check |(-rho) - lambda_i| > 0, equivalently |rho + lambda_i| > 0.
    gap1 = tf.abs(rho + lam1)  # Note: rho + lambda is also the NEC margin!
    gap2 = tf.abs(rho + lam2)
    gap3 = tf.abs(rho + lam3)
    eigenvalue_gap = tf.minimum(gap1, tf.minimum(gap2, gap3))

    # Type I classification: flux small AND eigenvalue gap positive
    tol_flux = tf.constant(1e-6, dtype=tf.float32)
    tol_gap = tf.constant(1e-12, dtype=tf.float32)
    is_type1 = tf.logical_and(flux_norm < tol_flux, eigenvalue_gap > tol_gap)

    type1_fraction = tf.reduce_mean(tf.cast(is_type1, tf.float32))

    return is_type1, flux_norm, eigenvalue_gap, type1_fraction
