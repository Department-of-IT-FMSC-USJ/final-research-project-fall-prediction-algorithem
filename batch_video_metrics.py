# batch_video_metrics.py
"""Batch-process a directory of videos, extracting fall-detection metrics.

This small wrapper iterates over every video file in a directory, calls the
`video_metrics._process_video` helper for each one, and saves the per-frame
metrics JSON next to the existing samples in *metrics_json/*.  The logic and
thresholds remain **exactly** the same as for the single-file version because we
reuse the internal helpers from ``video_metrics.py``.

Example
-------
Process the Coffee-room sample set provided by the user:

>>> python batch_video_metrics.py --dir "C:\Users\kasun\OneDrive - Faculty of Management Studies and Commerce\BIS\Research Docs\ML video sampels\Coffee_room_01\Coffee_room_01\Videos" --pattern "*.avi"

Use ``--show`` to preview the annotated frames during processing (slower).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

# Reuse the processing logic from the single-file script
from video_metrics import _process_video, _ensure_output_path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Batch fall-detection metrics extraction for all videos in a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        default=r"C:\Users\kasun\OneDrive - Faculty of Management Studies and Commerce\BIS\Research Docs\ML video sampels\Coffee_room_01\Coffee_room_01\Videos",
        help="Directory containing videos to analyse (default: Coffee_room_01 sample set).",
    )
    parser.add_argument(
        "--pattern",
        default="*.mp4",
        help="Glob pattern for video files (e.g. '*.mp4' or '*.avi').",
    )
    parser.add_argument(
        "--pose-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe Pose complexity passed through to video_metrics.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show annotated preview while processing (slower).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Directory processing helper
# ---------------------------------------------------------------------------

def _iter_videos(directory: Path, pattern: str) -> Iterable[Path]:
    """Yield all files in *directory* matching *pattern* (non-recursive)."""
    yield from directory.glob(pattern)


def process_directory(
    directory: Path,
    pattern: str = "*.mp4",
    pose_complexity: int = 1,
    show: bool = False,
) -> None:
    """Run fall-detection analysis on every video in *directory*."""

    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    videos = list(_iter_videos(directory, pattern))
    if not videos:
        print(f"No videos matching '{pattern}' found in '{directory}'.")
        return

    print(f"Found {len(videos)} video(s) – starting analysis…\n")

    for idx, vid_path in enumerate(videos, 1):
        print(f"[{idx}/{len(videos)}] Processing '{vid_path.name}' …")
        out_path = _ensure_output_path(vid_path, None)
        _process_video(vid_path, out_path, pose_complexity=pose_complexity, show=show)
        print()

    print("All videos processed.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    args = _parse_args()
    process_directory(
        Path(args.dir).expanduser(),
        pattern=args.pattern,
        pose_complexity=args.pose_complexity,
        show=args.show,
    )


if __name__ == "__main__":
    main() 