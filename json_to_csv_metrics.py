# json_to_csv_metrics.py
"""Convert per-frame metrics JSON files to CSV.

This utility scans a directory of JSON files (default: *metrics_json/*), each
containing a list of per-frame metrics produced by *video_metrics.py*, and
writes a corresponding CSV to *metrics_csv/<same-stem>_metrics.csv*.

Usage
-----
Run with defaults:

    python json_to_csv_metrics.py

Specify alternative folders or overwrite behaviour:

    python json_to_csv_metrics.py --json-dir other_json --csv-dir out_csv --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd  # type: ignore


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Convert *_metrics.json files to CSV format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json-dir",
        default="metrics_json",
        help="Directory containing JSON metrics files.",
    )
    parser.add_argument(
        "--csv-dir",
        default="metrics_csv",
        help="Destination directory for CSV outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files if they already exist.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------

def _convert_file(json_path: Path, csv_path: Path, overwrite: bool = False) -> None:
    if csv_path.exists() and not overwrite:
        print(f"[SKIP] {csv_path.name} already exists. Use --overwrite to regenerate.")
        return

    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse '{json_path}': {e}")
        return

    if not isinstance(data, list):
        print(f"[ERROR] Unexpected JSON structure in '{json_path}' – expected a list.")
        return

    # Insert frame index and maybe video id for clarity
    for idx, entry in enumerate(data):
        if isinstance(entry, dict):
            entry.setdefault("frame", idx)
        else:
            print(f"[WARN] Non-dict entry at index {idx} in {json_path.name}; skipping.")
            data[idx] = {}

    df = pd.DataFrame(data)
    # Ensure consistent column order
    cols = [
        "frame",
        "trunk_angle",
        "nsar",
        "theta_u",
        "theta_d",
        "fall_detected",
    ]
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved {len(df)} rows to '{csv_path}'.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    args = _parse_args()
    json_dir = Path(args.json_dir).expanduser().resolve()
    csv_dir = Path(args.csv_dir).expanduser().resolve()

    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory '{json_dir}' does not exist.")

    json_files = sorted(json_dir.glob("*_metrics.json"))
    if not json_files:
        print(f"No *_metrics.json files found in '{json_dir}'.")
        return

    print(f"Converting {len(json_files)} file(s)…\n")

    for jp in json_files:
        cp = csv_dir / f"{jp.stem}.csv"
        _convert_file(jp, cp, overwrite=args.overwrite)

    print("\nAll conversions complete.")


if __name__ == "__main__":
    main() 