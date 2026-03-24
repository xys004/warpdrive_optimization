"""
Render high-resolution energy-condition plane maps from .npy arrays.
Produces publication-quality PNGs matching the Mathematica notebook style.

Usage:
    python render_hires_maps.py --domain 2 --input-dir hires_maps/domain2 --output-dir figures/hires_python_domain2
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── colormap matching Mathematica's diverging palette ─────────────────────────
def make_diverging_cmap():
    """Blue-white-red diverging colormap (matches Mathematica BlueGreenYellow-ish)."""
    return plt.cm.RdBu_r   # red=positive, blue=negative, white=zero

FIELD_META = {
    "rho":        {"label": r"Energy density $\rho$",          "symmetric": False},
    "WEC_min":    {"label": r"WEC min: $\rho + \min(\lambda,0)$", "symmetric": True},
    "NEC_min":    {"label": r"NEC min: $\rho + \min\lambda$",   "symmetric": True},
    "DEC_margin": {"label": r"DEC margin: $\rho - |\lambda|_{\max}$", "symmetric": True},
    "SEC":        {"label": r"SEC: $\rho + \Sigma\lambda$",     "symmetric": True},
}

def render_plane(npy_dir, plane, domain, out_dir, dpi=300):
    """Load and render all fields for one plane (XZ or XY)."""
    os.makedirs(out_dir, exist_ok=True)

    # Load axes
    ax_x = np.load(os.path.join(npy_dir, f"domain{domain}_{plane}_axis_x.npy"))
    ax_y = np.load(os.path.join(npy_dir, f"domain{domain}_{plane}_axis_y.npy"))

    with open(os.path.join(npy_dir, f"domain{domain}_{plane}_meta.json")) as f:
        meta = json.load(f)
    xlabel, ylabel = meta["xlabel"], meta["ylabel"]
    params = meta["parameters"]

    tag = f"domain{domain}_{plane}"
    cmap = make_diverging_cmap()

    for field_name, fmeta in FIELD_META.items():
        fpath = os.path.join(npy_dir, f"domain{domain}_{plane}_{field_name}_n{len(ax_x)}.npy")
        if not os.path.exists(fpath):
            print(f"  [skip] {fpath}")
            continue

        data = np.load(fpath)

        # Clip extreme outliers (top/bottom 1%) for display
        valid = data[np.isfinite(data)]
        lo, hi = np.nanpercentile(valid, 1), np.nanpercentile(valid, 99)

        if fmeta["symmetric"]:
            vmax = max(abs(lo), abs(hi))
            vmin = -vmax
        else:
            vmin, vmax = lo, hi

        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) \
               if fmeta["symmetric"] and vmin < 0 < vmax \
               else mcolors.Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(
            data.T,
            origin="lower",
            extent=[ax_x[0], ax_x[-1], ax_y[0], ax_y[-1]],
            cmap=cmap,
            norm=norm,
            interpolation="bicubic",
            aspect="equal",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(xlabel.upper())
        ax.set_ylabel(ylabel.upper())
        ax.set_title(fmeta["label"], fontsize=10)

        # Add shell radius lines
        if params.get("r_inner") and params["r_inner"] > 0:
            for r in [params["r_inner"], params["r_outer"]]:
                circle = plt.Circle((0, 0), r, color="white", fill=False,
                                    linestyle="--", linewidth=0.8, alpha=0.7)
                ax.add_patch(circle)

        fig.tight_layout()
        out_name = f"domain{domain}_{plane}_{field_name}_hires.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",     type=int, default=2)
    parser.add_argument("--input-dir",  default="hires_maps/domain2")
    parser.add_argument("--output-dir", default="figures/hires_python")
    parser.add_argument("--dpi",        type=int, default=300)
    args = parser.parse_args()

    for plane in ("XZ", "XY"):
        print(f"\n[Rendering] Domain {args.domain} — plane {plane}")
        render_plane(args.input_dir, plane, args.domain, args.output_dir, dpi=args.dpi)

    print("\nDone.")

if __name__ == "__main__":
    main()
