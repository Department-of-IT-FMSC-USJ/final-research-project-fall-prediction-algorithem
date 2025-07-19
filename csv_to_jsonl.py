#!/usr/bin/env python3
"""
csv_to_jsonl.py

Optimized utility to convert a CSV dataset to JSON Lines (JSONL)
format that can be ingested by generative AI fine-tuning pipelines.

Features
--------
1. Streaming/Chunked processing with pandas to keep memory footprint low
2. Robust CLI with custom column names
3. Per-row JSON validation; skips malformed entries
4. Graceful handling of missing files and empty values
5. Progress indicators and a final summary report

Usage
-----
python csv_to_jsonl.py \
       --input data.csv \
       --output metrics.jsonl \
       --prompt_col prompt \
       --completion_col completion
"""
import argparse
import json
import os
import sys
import random
from typing import Any, Dict, List

import pandas as pd

SYSTEM_PROMPT = "You are a fall detection algorithm"

# Message templates
FALL_TEMPLATES: List[str] = [
    "⚠️  Imminent fall detected. Trunk angle reached {trunk_angle}°, NSAR at {nsar}." \
    " Take immediate precaution! Cause: rapid forward lean and unstable stance.",
    "Alert! Metrics show trunk angle {trunk_angle}° and downward tilt {theta_d}°." \
    " High probability of fall—recommend holding onto support.",
    "Red flag: balance deviation detected (NSAR={nsar}). You may fall soon if posture" \
    " is not corrected.",
]

SAFE_TEMPLATES: List[str] = [
    "✅ Posture looks stable. All metrics within safe range.",
    "Balance nominal. Trunk angle at {trunk_angle}°, well below risk threshold.",
    "👌 No fall risk detected this frame. Keep up the steady stance!",
]

# Default feature columns to include in auto prompt generation
DEFAULT_FEATURE_COLS = ["trunk_angle", "nsar", "theta_u", "theta_d"]


def validate_record(record: Dict[str, Any]) -> bool:
    """Validate that *record* is JSON serializable and contains non-empty user content."""
    try:
        json.dumps(record)
    except (TypeError, ValueError):
        return False

    # Attempt to extract user message content
    try:
        user_val = record["messages"][1]["content"]
    except (KeyError, IndexError, TypeError):
        user_val = ""

    if user_val is None or str(user_val).strip() == "":
        return False
    return True


def process_csv(
    input_path: str,
    output_path: str,
    prompt_col: str = "prompt",
    completion_col: str = "completion",
    chunk_size: int = 10_000,
    feature_cols: List[str] | None = None,
    seed: int | None = None,
) -> None:
    """Convert *input_path* CSV to JSONL saved at *output_path*."""

    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Validate required columns exist before streaming
    try:
        header_cols = list(pd.read_csv(input_path, nrows=0).columns)
    except Exception as err:
        print(f"❌ Failed to read CSV header: {err}", file=sys.stderr)
        sys.exit(1)

    prompt_available = prompt_col in header_cols

    # Use provided feature columns or defaults that exist in file
    if feature_cols is None:
        feature_cols = [col for col in DEFAULT_FEATURE_COLS if col in header_cols]

    if not prompt_available and not feature_cols:
        print(
            "❌ No prompt column available and none of the default feature columns are present to generate a prompt.",
            file=sys.stderr,
        )
        sys.exit(1)

    if seed is not None:
        random.seed(seed)

    processed = 0
    written = 0
    skipped = 0

    # Open once for efficiency
    with open(output_path, "w", encoding="utf-8") as fout:
        try:
            reader = pd.read_csv(input_path, chunksize=chunk_size, dtype=str)
        except Exception as err:
            print(f"❌ Failed to read CSV: {err}", file=sys.stderr)
            sys.exit(1)

        for chunk_idx, chunk in enumerate(reader, start=1):
            for row in chunk.itertuples(index=False):
                processed += 1

                # Build user prompt
                if prompt_available:
                    prompt_val = getattr(row, prompt_col)
                else:
                    # Auto-create prompt from feature columns present
                    prompt_parts = [f"{col}={getattr(row, col, '?')}" for col in feature_cols]
                    prompt_val = "Metrics: " + ", ".join(prompt_parts) + ". Predict fall risk."

                # Determine fall detection status
                fall_raw = getattr(row, "fall_detected", "0")
                fall_bool = str(fall_raw).strip().lower() in {"true", "1", "yes", "y", "t"}

                template_source = FALL_TEMPLATES if fall_bool else SAFE_TEMPLATES
                template = random.choice(template_source)

                # Fill placeholders if present in template
                assistant_val = template.format(
                    trunk_angle=getattr(row, "trunk_angle", "?"),
                    nsar=getattr(row, "nsar", "?"),
                    theta_u=getattr(row, "theta_u", "?"),
                    theta_d=getattr(row, "theta_d", "?"),
                )

                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_val},
                        {"role": "assistant", "content": assistant_val},
                    ]
                }

                if validate_record(record):
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                else:
                    skipped += 1

            # Print periodic progress
            if processed % chunk_size == 0:
                print(
                    f"Processed {processed} rows | Written: {written} | Skipped: {skipped}",
                    file=sys.stderr,
                )

    # Final summary
    print(
        f"✅ Conversion complete. Total processed: {processed}, Written: {written}, Skipped: {skipped}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a CSV file to JSONL with system/user/assistant keys.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input CSV file")
    parser.add_argument(
        "--output",
        "-o",
        default="metrcs.jsonl",
        help="Path for the output JSONL file",
    )
    parser.add_argument(
        "--prompt_col", default="prompt", help="Column name containing the user prompt"
    )
    parser.add_argument(
        "--completion_col",
        default="completion",
        help="Column name containing the completion (not used but kept for compatibility)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10_000,
        help="Number of rows to process per chunk",
    )
    parser.add_argument(
        "--features",
        help="Comma-separated list of feature columns to include in auto-generated prompt when --prompt_col is missing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible creative assistant messages",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_csv(
        input_path=args.input,
        output_path=args.output,
        prompt_col=args.prompt_col,
        completion_col=args.completion_col,
        chunk_size=args.chunk_size,
        feature_cols=[c.strip() for c in args.features.split(",")] if args.features else None,
        seed=args.seed,
    ) 