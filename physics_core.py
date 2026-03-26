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
