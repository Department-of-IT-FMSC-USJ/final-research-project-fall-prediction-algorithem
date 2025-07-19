# combine_json_metrics.py
"""Merge all per-frame JSON metrics into a single CSV.

This tool reads every ``*_metrics.json`` in *metrics_json/*, adds the video
filename (stem) and frame index to each record, concatenates them, and writes a
master CSV to *metrics_csv/all_metrics.csv* (or a custom path via ``--out``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd  # type: ignore


def _parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser(
        description="Combine JSON metrics into a single CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--json-dir",
        default="metrics_json",
        help="Directory containing *_metrics.json files.",
    )
    p.add_argument(
        "--out",
        default="metrics_csv/all_metrics.csv",
        help="Output CSV file path.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite CSV if it exists.",
    )
    return p.parse_args()


def _load_json(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("Top-level JSON is not a list")
        return data
    except Exception as e:
        print(f"[ERROR] Can't parse {path}: {e}")
        return []


def main() -> None:  # noqa: D401
    args = _parse_args()
    json_dir = Path(args.json_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory '{json_dir}' not found")

    files = sorted(json_dir.glob("*_metrics.json"))
    if not files:
        print("No JSON metrics found – nothing to do.")
        return

    rows: List[pd.DataFrame] = []

    for jp in files:
        video_id = jp.stem.replace("_metrics", "")
        frames = _load_json(jp)
        if not frames:
            continue
        # ensure each record is dict and add frame index
        for idx, rec in enumerate(frames):
            if not isinstance(rec, dict):
                continue
            rec.setdefault("frame", idx)
        df = pd.DataFrame(frames)
        df["video"] = video_id
        rows.append(df)
        print(f"Loaded {len(df)} frames from {jp.name}")

    if not rows:
        print("No valid data to combine.")
        return

    combined = pd.concat(rows, ignore_index=True, sort=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        print(f"Output {out_path} exists. Use --overwrite to replace.")
        return

    combined.to_csv(out_path, index=False)
    print(f"Combined {len(combined)} rows into '{out_path}'.")


if __name__ == "__main__":
    main() 