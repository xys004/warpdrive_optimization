from __future__ import annotations

"""Build a release manifest for a referee-facing code/data archive.

This helper walks a curated set of files and directories, records file sizes and SHA256 hashes,
and writes both JSON and Markdown manifests. The goal is to make a Zenodo or journal-companion
archive self-describing: referees should be able to see exactly which bundles, plots, scripts,
and manuscript-support files belong to a given release.

The script is intentionally conservative:
- it ignores paths that do not exist,
- it only records regular files,
- it never mutates the release payload itself,
- and it keeps all output under a dedicated manifest directory.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

DEFAULT_PATHS = [
    "README.md",
    "requirements.txt",
    "LICENSE",
    "CITATION.cff",
    "physics_core.py",
    "einstein_optimizer.py",
    "postprocess_plots.py",
    "verify_outputs.py",
    "generate_run_bundle.py",
    "run_golden_dataset.py",
    "evaluate_manuscript_targets.py",
    "golden_dataset",
    "manuscript_target_bundles",
    "phase_c_domain1_audit.md",
    "phase_c_domain2_audit.md",
    "phase_c_synthesis.md",
    "manuscript_revision_explained.tex",
    "manuscript_revision_map.tex",
    "referee_revision_summary.tex",
]


@dataclass
class ManifestEntry:
    relative_path: str
    size_bytes: int
    sha256: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path, requested_paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for requested in requested_paths:
        path = (root / requested).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            raise ValueError(f"Requested path escapes repository root: {requested}")
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            files.append(child)
    return sorted({p for p in files})


def build_manifest(root: Path, requested_paths: Iterable[str]) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    for path in iter_files(root, requested_paths):
        entries.append(
            ManifestEntry(
                relative_path=path.relative_to(root).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return entries


def write_json(entries: list[ManifestEntry], output_path: Path) -> None:
    payload = {
        "entry_count": len(entries),
        "entries": [asdict(entry) for entry in entries],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(entries: list[ManifestEntry], output_path: Path) -> None:
    lines = [
        "# Release Manifest",
        "",
        f"Entries: {len(entries)}",
        "",
        "| Path | Size (bytes) | SHA256 |",
        "|---|---:|---|",
    ]
    for entry in entries:
        lines.append(f"| `{entry.relative_path}` | {entry.size_bytes} | `{entry.sha256}` |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate JSON and Markdown manifests for a release payload.")
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        help="File or directory to include. May be passed multiple times. Defaults to the curated release set.",
    )
    parser.add_argument(
        "--outdir",
        default="release_artifacts",
        help="Directory where manifest files will be written.",
    )
    parser.add_argument(
        "--stem",
        default="release_manifest",
        help="Base name for the generated manifest files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    requested_paths = args.paths if args.paths else DEFAULT_PATHS
    entries = build_manifest(root, requested_paths)

    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"{args.stem}.json"
    markdown_path = outdir / f"{args.stem}.md"

    write_json(entries, json_path)
    write_markdown(entries, markdown_path)

    print(f"Wrote {len(entries)} entries to:")
    print(f"  {json_path}")
    print(f"  {markdown_path}")


if __name__ == "__main__":
    main()
