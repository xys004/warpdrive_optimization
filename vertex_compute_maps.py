"""
Vertex AI job: compute high-resolution energy-condition plane maps.

Usage (local test):
    python vertex_compute_maps.py --domain 2 --plane-n 2000 \
        --input-dir ./golden_dataset --output-dir ./out

Usage (Vertex AI Training Job):
    Set env vars AIP_MODEL_DIR (GCS output path) automatically injected by Vertex.
    Pass --input-gcs gs://warpopt-data/golden_dataset/
         --output-gcs gs://warpopt-data/hires_maps/
"""

import argparse
import json
import os
import sys
import numpy as np

# ── helpers ────────────────────────────────────────────────────────────────────

def _gcs_download(gcs_uri, local_dir):
    """Download a GCS prefix to local_dir using gsutil."""
    import subprocess
    os.makedirs(local_dir, exist_ok=True)
    subprocess.check_call(["gsutil", "-m", "cp", "-r", gcs_uri.rstrip("/") + "/*", local_dir])


def _gcs_upload(local_dir, gcs_uri):
    """Upload local_dir to GCS."""
    import subprocess
    subprocess.check_call(["gsutil", "-m", "cp", "-r", local_dir.rstrip("/") + "/*", gcs_uri])


def load_golden_run(input_dir, domain):
    """Load the golden-dataset run from a local directory."""
    sys.path.insert(0, input_dir)
    sys.path.insert(0, os.path.dirname(__file__))

    # Find base tag
    import glob
    metas = glob.glob(os.path.join(input_dir, f"domain_{domain}_*_metadata.json"))
    if not metas:
        raise FileNotFoundError(f"No metadata JSON for domain {domain} in {input_dir}")
    meta_path = metas[0]

    with open(meta_path) as f:
        metadata = json.load(f)

    base = os.path.splitext(meta_path)[0].replace("_metadata", "")

    # final_params
    fp_path = base + "_final_params.csv"
    params = {}
    with open(fp_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    header = lines[0].split(",")
    vals   = lines[1].split(",")
    for k, v in zip(header, vals):
        v = v.strip()
        if v == "":
            params[k.strip()] = float("nan")  # e.g. R0 absent in domain 1
        else:
            try:    params[k.strip()] = float(v)
            except: params[k.strip()] = v

    return {
        "metadata":    metadata,
        "final_params": params,
        "losses":      None,
        "parameters":  None,
        "success_rates": None,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",     type=int, default=2,
                        help="1 = single-shell, 2 = double-shell")
    parser.add_argument("--plane-n",    type=int, default=2000,
                        help="Points per side of the 2-D plane grid")
    parser.add_argument("--input-dir",  default="./golden_dataset",
                        help="Local directory with golden-dataset CSV/JSON files")
    parser.add_argument("--output-dir", default="./hires_maps",
                        help="Local directory to write .npy output files")
    parser.add_argument("--input-gcs",  default=None,
                        help="GCS URI to download input data from (optional)")
    parser.add_argument("--output-gcs", default=None,
                        help="GCS URI to upload results to (optional)")
    args = parser.parse_args()

    # ── download input from GCS if requested ───────────────────────────────────
    if args.input_gcs:
        print(f"[GCS] Downloading input from {args.input_gcs} ...")
        _gcs_download(args.input_gcs, args.input_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load run ────────────────────────────────────────────────────────────────
    print(f"[INFO] Loading domain {args.domain} golden run from {args.input_dir}")
    run = load_golden_run(args.input_dir, args.domain)
    p = run["final_params"]
    print(f"[INFO] Parameters: A={p.get('A')}, B={p.get('B')}, R0={p.get('R0','—')}, alpha={p.get('alpha','—')}")
    print(f"[INFO] Plane resolution: {args.plane_n} x {args.plane_n} = {args.plane_n**2:,} points/plane")

    # ── compute planes ──────────────────────────────────────────────────────────
    # Import here so TF initialises after argument parsing
    from postprocess_plots import compute_plane_map

    for plane in ("XZ", "XY"):
        print(f"\n[COMPUTE] Plane {plane} ...")
        result = compute_plane_map(run, plane, plane_n=args.plane_n)

        # Save each field as a separate .npy + shared axis files
        for field_name, arr in result["fields"].items():
            out_name = f"domain{args.domain}_{plane}_{field_name}_n{args.plane_n}.npy"
            out_path = os.path.join(args.output_dir, out_name)
            np.save(out_path, arr.astype(np.float32))
            print(f"  -> {out_path}  shape={arr.shape}  "
                  f"min={np.nanmin(arr):.4g}  max={np.nanmax(arr):.4g}")

        # Save axis arrays once per plane
        np.save(os.path.join(args.output_dir, f"domain{args.domain}_{plane}_axis_x.npy"),
                result["axis_x"].astype(np.float32))
        np.save(os.path.join(args.output_dir, f"domain{args.domain}_{plane}_axis_y.npy"),
                result["axis_y"].astype(np.float32))

        # Save a small JSON with parameters / axis labels
        meta_out = {
            "plane":      result["plane"],
            "xlabel":     result["xlabel"],
            "ylabel":     result["ylabel"],
            "plane_n":    args.plane_n,
            "parameters": result["parameters"],
        }
        with open(os.path.join(args.output_dir,
                               f"domain{args.domain}_{plane}_meta.json"), "w") as f:
            json.dump(meta_out, f, indent=2)

    print("\n[DONE] All planes computed.")

    # ── upload output to GCS if requested ──────────────────────────────────────
    if args.output_gcs:
        print(f"[GCS] Uploading results to {args.output_gcs} ...")
        _gcs_upload(args.output_dir, args.output_gcs)
        print("[GCS] Upload complete.")


if __name__ == "__main__":
    main()
