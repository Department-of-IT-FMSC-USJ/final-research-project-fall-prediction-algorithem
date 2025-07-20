#!/usr/bin/env python3
"""
Main launcher for the Fall Prediction System.

This script provides a unified interface to all the refactored functionality:
- Live webcam fall detection
- Video processing and analysis
- Batch processing
- Data conversion
- Metrics analysis
- Flask web application
"""

import argparse
import sys
from pathlib import Path

from utils.video_processor import VideoProcessor
from utils.batch_processor import BatchProcessor
from utils.data_converter import DataConverter
from utils.metrics_analyzer import MetricsAnalyzer


def main():
    """Main entry point for the Fall Prediction System."""
    parser = argparse.ArgumentParser(
        description="Fall Prediction System - Unified Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Available modes')
    
    # Live detection mode
    live_parser = subparsers.add_parser('live', help='Live webcam fall detection')
    live_parser.add_argument(
        "--sources",
        type=str,
        default="0",
        help="Comma-separated list of video sources",
    )
    live_parser.add_argument(
        "--multi-view",
        action="store_true",
        help="Enable multi-camera processing",
    )
    live_parser.add_argument(
        "--pose-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe Pose complexity",
    )
    
    # Video processing mode
    video_parser = subparsers.add_parser('video', help='Process single video file')
    video_parser.add_argument("--input", "-i", required=True, help="Input video file")
    video_parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file (default: metrics_json/<input-stem>_metrics.json)",
    )
    video_parser.add_argument(
        "--pose-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe Pose complexity",
    )
    video_parser.add_argument(
        "--show",
        action="store_true",
        help="Show preview during processing",
    )
    
    # Batch processing mode
    batch_parser = subparsers.add_parser('batch', help='Process multiple videos')
    batch_parser.add_argument(
        "--dir",
        default=".",
        help="Directory containing videos",
    )
    batch_parser.add_argument(
        "--pattern",
        default="*.mp4",
        help="File pattern for videos",
    )
    batch_parser.add_argument(
        "--pose-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe Pose complexity",
    )
    batch_parser.add_argument(
        "--show",
        action="store_true",
        help="Show preview during processing",
    )
    
    # Data conversion mode
    convert_parser = subparsers.add_parser('convert', help='Convert CSV to JSONL')
    convert_parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    convert_parser.add_argument(
        "--output",
        "-o",
        default="metrics.jsonl",
        help="Output JSONL file",
    )
    convert_parser.add_argument(
        "--prompt_col",
        default="prompt",
        help="Column name for prompts",
    )
    convert_parser.add_argument(
        "--chunk_size",
        type=int,
        default=10_000,
        help="Chunk size for processing",
    )
    
    # Analysis mode
    analyze_parser = subparsers.add_parser('analyze', help='Analyze metrics')
    analyze_parser.add_argument("--input", "-i", required=True, help="Input JSON metrics file")
    analyze_parser.add_argument(
        "--output",
        "-o",
        help="Output directory for plots and reports",
    )
    analyze_parser.add_argument(
        "--report",
        action="store_true",
        help="Generate summary report",
    )
    analyze_parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate visualization plots",
    )
    
    # Web app mode
    web_parser = subparsers.add_parser('web', help='Launch Flask web application')
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to",
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to",
    )
    web_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return 1
    
    try:
        if args.mode == 'live':
            return run_live_detection(args)
        elif args.mode == 'video':
            return run_video_processing(args)
        elif args.mode == 'batch':
            return run_batch_processing(args)
        elif args.mode == 'convert':
            return run_data_conversion(args)
        elif args.mode == 'analyze':
            return run_metrics_analysis(args)
        elif args.mode == 'web':
            return run_web_application(args)
        else:
            print(f"Unknown mode: {args.mode}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def run_live_detection(args):
    """Run live webcam fall detection."""
    print("Starting live fall detection...")
    
    # Import here to avoid circular imports
    from webcam_feed_refactored import main as live_main
    
    # Set up sys.argv for the live detection script
    sys.argv = [
        'webcam_feed_refactored.py',
        '--sources', args.sources,
        '--pose-complexity', str(args.pose_complexity)
    ]
    if args.multi_view:
        sys.argv.append('--multi-view')
    
    live_main()
    return 0


def run_video_processing(args):
    """Run single video processing."""
    print(f"Processing video: {args.input}")
    
    processor = VideoProcessor(args.pose_complexity, args.show)
    try:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else None
        
        processor.process_video(input_path, output_path)
        print("✅ Video processing completed successfully!")
        return 0
    finally:
        processor.cleanup()


def run_batch_processing(args):
    """Run batch video processing."""
    print(f"Processing videos in directory: {args.dir}")
    
    processor = BatchProcessor(args.pose_complexity, args.show)
    try:
        directory = Path(args.dir).expanduser()
        processor.process_directory(directory, args.pattern)
        return 0
    finally:
        processor.cleanup()


def run_data_conversion(args):
    """Run data conversion."""
    print(f"Converting CSV to JSONL: {args.input}")
    
    converter = DataConverter()
    stats = converter.csv_to_jsonl(
        args.input,
        args.output,
        args.prompt_col,
        chunk_size=args.chunk_size
    )
    
    print(
        f"✅ Conversion complete. Total processed: {stats['processed']}, "
        f"Written: {stats['written']}, Skipped: {stats['skipped']}"
    )
    return 0


def run_metrics_analysis(args):
    """Run metrics analysis."""
    print(f"Analyzing metrics: {args.input}")
    
    analyzer = MetricsAnalyzer()
    df = analyzer.load_metrics(args.input)
    print(f"Loaded {len(df)} frames from {args.input}")
    
    if args.report:
        report = analyzer.create_summary_report(df)
        print(report)
    
    if args.plots:
        output_path = None
        if args.output:
            output_path = Path(args.output) / f"{Path(args.input).stem}_analysis.png"
        analyzer.plot_metrics(df, output_path)
    
    return 0


def run_web_application(args):
    """Run Flask web application."""
    print(f"Starting Flask web application on {args.host}:{args.port}")
    
    # Import here to avoid circular imports
    import subprocess
    import os
    
    # Change to the app directory
    app_dir = Path(__file__).parent / "fall_prediction_app"
    if not app_dir.exists():
        print(f"Error: Application directory not found at {app_dir}")
        return 1
    
    os.chdir(app_dir)
    
    # Build command
    cmd = [
        sys.executable, "app.py",
        "--host", args.host,
        "--port", str(args.port)
    ]
    if args.debug:
        cmd.append("--debug")
    
    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running web application: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 