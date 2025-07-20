"""Batch processing utilities for multiple videos."""

import argparse
from pathlib import Path
from typing import Iterable, List
from .video_processor import VideoProcessor


class BatchProcessor:
    """Handles batch processing of multiple video files."""
    
    def __init__(self, pose_complexity: int = 1, show_preview: bool = False):
        """Initialize batch processor.
        
        Args:
            pose_complexity: MediaPipe pose complexity
            show_preview: Whether to show preview during processing
        """
        self.pose_complexity = pose_complexity
        self.show_preview = show_preview
        self.video_processor = VideoProcessor(pose_complexity, show_preview)
    
    def process_directory(self, directory: Path, pattern: str = "*.mp4") -> List[Path]:
        """Process all videos in a directory.
        
        Args:
            directory: Directory containing videos
            pattern: File pattern to match (e.g., "*.mp4", "*.avi")
            
        Returns:
            List of processed video paths
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory '{directory}' does not exist.")
        
        videos = list(self._iter_videos(directory, pattern))
        if not videos:
            print(f"No videos matching '{pattern}' found in '{directory}'.")
            return []
        
        print(f"Found {len(videos)} video(s) – starting analysis…\n")
        
        processed_videos = []
        
        for idx, vid_path in enumerate(videos, 1):
            try:
                print(f"[{idx}/{len(videos)}] Processing '{vid_path.name}' …")
                self.video_processor.process_video(vid_path)
                processed_videos.append(vid_path)
                print(f"✅ Completed: {vid_path.name}")
            except Exception as e:
                print(f"❌ Failed to process {vid_path.name}: {e}")
            print()
        
        print(f"Batch processing complete. {len(processed_videos)}/{len(videos)} videos processed successfully.")
        return processed_videos
    
    def _iter_videos(self, directory: Path, pattern: str) -> Iterable[Path]:
        """Yield all video files in directory matching pattern.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            
        Yields:
            Video file paths
        """
        yield from directory.glob(pattern)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.video_processor.cleanup()


def main():
    """Command-line interface for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch fall-detection metrics extraction for all videos in a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory containing videos to analyse.",
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
        help="MediaPipe Pose complexity passed through to video_processor.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show annotated preview while processing (slower).",
    )
    
    args = parser.parse_args()
    
    # Process directory
    processor = BatchProcessor(args.pose_complexity, args.show)
    try:
        directory = Path(args.dir).expanduser()
        processor.process_directory(directory, args.pattern)
    finally:
        processor.cleanup()


if __name__ == "__main__":
    main() 