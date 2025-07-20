"""Metrics analysis and visualization utilities."""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


class MetricsAnalyzer:
    """Analyzes and visualizes fall detection metrics."""
    
    def __init__(self):
        """Initialize metrics analyzer."""
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def load_metrics(self, file_path: str) -> pd.DataFrame:
        """Load metrics from JSON file into DataFrame.
        
        Args:
            file_path: Path to JSON metrics file
            
        Returns:
            DataFrame with metrics
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        return df
    
    def analyze_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze metrics and return statistics.
        
        Args:
            df: DataFrame with metrics
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['total_frames'] = len(df)
        analysis['fall_detected_frames'] = df.get('fall_detected', pd.Series()).sum()
        analysis['fall_detection_rate'] = analysis['fall_detected_frames'] / analysis['total_frames'] if analysis['total_frames'] > 0 else 0
        
        # Metrics statistics
        for metric in ['trunk_angle', 'nsar', 'theta_u', 'theta_d']:
            if metric in df.columns:
                metric_data = pd.to_numeric(df[metric], errors='coerce').dropna()
                if len(metric_data) > 0:
                    analysis[f'{metric}_mean'] = metric_data.mean()
                    analysis[f'{metric}_std'] = metric_data.std()
                    analysis[f'{metric}_min'] = metric_data.min()
                    analysis[f'{metric}_max'] = metric_data.max()
                    analysis[f'{metric}_median'] = metric_data.median()
        
        return analysis
    
    def plot_metrics(self, df: pd.DataFrame, output_path: Optional[str] = None, 
                    show_plots: bool = True) -> None:
        """Create visualization plots for metrics.
        
        Args:
            df: DataFrame with metrics
            output_path: Optional path to save plots
            show_plots: Whether to display plots
        """
        # Convert metrics to numeric
        metrics_df = df.copy()
        for col in ['trunk_angle', 'nsar', 'theta_u', 'theta_d']:
            if col in metrics_df.columns:
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Fall Detection Metrics Analysis', fontsize=16)
        
        # Plot 1: Trunk Angle over time
        if 'trunk_angle' in metrics_df.columns:
            axes[0, 0].plot(metrics_df.index, metrics_df['trunk_angle'], 'b-', alpha=0.7)
            axes[0, 0].set_title('Trunk Angle Over Time')
            axes[0, 0].set_xlabel('Frame')
            axes[0, 0].set_ylabel('Trunk Angle (degrees)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: NSAR over time
        if 'nsar' in metrics_df.columns:
            axes[0, 1].plot(metrics_df.index, metrics_df['nsar'], 'g-', alpha=0.7)
            axes[0, 1].set_title('NSAR Over Time')
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('NSAR')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Plumb angles
        if 'theta_u' in metrics_df.columns and 'theta_d' in metrics_df.columns:
            axes[1, 0].plot(metrics_df.index, metrics_df['theta_u'], 'r-', alpha=0.7, label='θu')
            axes[1, 0].plot(metrics_df.index, metrics_df['theta_d'], 'm-', alpha=0.7, label='θd')
            axes[1, 0].set_title('Plumb Angles Over Time')
            axes[1, 0].set_xlabel('Frame')
            axes[1, 0].set_ylabel('Angle (degrees)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Fall detection events
        if 'fall_detected' in metrics_df.columns:
            fall_frames = metrics_df[metrics_df['fall_detected'] == True].index
            axes[1, 1].scatter(fall_frames, [1] * len(fall_frames), c='red', s=50, alpha=0.7)
            axes[1, 1].set_title('Fall Detection Events')
            axes[1, 1].set_xlabel('Frame')
            axes[1, 1].set_ylabel('Fall Detected')
            axes[1, 1].set_ylim(0, 2)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {output_path}")
        
        if show_plots:
            plt.show()
    
    def create_summary_report(self, df: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """Create a summary report of the metrics analysis.
        
        Args:
            df: DataFrame with metrics
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        analysis = self.analyze_metrics(df)
        
        report = []
        report.append("=" * 60)
        report.append("FALL DETECTION METRICS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic statistics
        report.append("BASIC STATISTICS:")
        report.append(f"  Total frames analyzed: {analysis['total_frames']}")
        report.append(f"  Frames with fall detected: {analysis['fall_detected_frames']}")
        report.append(f"  Fall detection rate: {analysis['fall_detection_rate']:.2%}")
        report.append("")
        
        # Metrics statistics
        report.append("METRICS STATISTICS:")
        for metric in ['trunk_angle', 'nsar', 'theta_u', 'theta_d']:
            if f'{metric}_mean' in analysis:
                report.append(f"  {metric.upper()}:")
                report.append(f"    Mean: {analysis[f'{metric}_mean']:.2f}")
                report.append(f"    Std:  {analysis[f'{metric}_std']:.2f}")
                report.append(f"    Min:  {analysis[f'{metric}_min']:.2f}")
                report.append(f"    Max:  {analysis[f'{metric}_max']:.2f}")
                report.append(f"    Median: {analysis[f'{metric}_median']:.2f}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_path}")
        
        return report_text
    
    def compare_videos(self, video_metrics: Dict[str, pd.DataFrame], 
                      output_path: Optional[str] = None) -> None:
        """Compare metrics across multiple videos.
        
        Args:
            video_metrics: Dictionary mapping video names to DataFrames
            output_path: Optional path to save comparison plots
        """
        if len(video_metrics) < 2:
            print("Need at least 2 videos for comparison")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Video Comparison - Fall Detection Metrics', fontsize=16)
        
        # Prepare data for comparison
        comparison_data = {}
        for video_name, df in video_metrics.items():
            for metric in ['trunk_angle', 'nsar', 'theta_u', 'theta_d']:
                if metric in df.columns:
                    key = f"{video_name}_{metric}"
                    comparison_data[key] = pd.to_numeric(df[metric], errors='coerce').dropna()
        
        # Plot 1: Box plot of trunk angles
        trunk_data = {name: data for name, data in comparison_data.items() if 'trunk_angle' in name}
        if trunk_data:
            axes[0, 0].boxplot(trunk_data.values(), labels=[name.replace('_trunk_angle', '') for name in trunk_data.keys()])
            axes[0, 0].set_title('Trunk Angle Distribution')
            axes[0, 0].set_ylabel('Trunk Angle (degrees)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Box plot of NSAR
        nsar_data = {name: data for name, data in comparison_data.items() if 'nsar' in name}
        if nsar_data:
            axes[0, 1].boxplot(nsar_data.values(), labels=[name.replace('_nsar', '') for name in nsar_data.keys()])
            axes[0, 1].set_title('NSAR Distribution')
            axes[0, 1].set_ylabel('NSAR')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Fall detection rates
        fall_rates = {}
        for video_name, df in video_metrics.items():
            if 'fall_detected' in df.columns:
                fall_rates[video_name] = df['fall_detected'].sum() / len(df)
        
        if fall_rates:
            video_names = list(fall_rates.keys())
            rates = list(fall_rates.values())
            axes[1, 0].bar(video_names, rates, color='red', alpha=0.7)
            axes[1, 0].set_title('Fall Detection Rates')
            axes[1, 0].set_ylabel('Fall Detection Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Average metrics comparison
        avg_metrics = {}
        for video_name, df in video_metrics.items():
            avg_metrics[video_name] = {}
            for metric in ['trunk_angle', 'nsar', 'theta_u', 'theta_d']:
                if metric in df.columns:
                    avg_metrics[video_name][metric] = pd.to_numeric(df[metric], errors='coerce').mean()
        
        if avg_metrics:
            metrics_names = list(avg_metrics[list(avg_metrics.keys())[0]].keys())
            video_names = list(avg_metrics.keys())
            
            x = np.arange(len(metrics_names))
            width = 0.8 / len(video_names)
            
            for i, video_name in enumerate(video_names):
                values = [avg_metrics[video_name].get(metric, 0) for metric in metrics_names]
                axes[1, 1].bar(x + i * width, values, width, label=video_name, alpha=0.7)
            
            axes[1, 1].set_title('Average Metrics Comparison')
            axes[1, 1].set_xlabel('Metrics')
            axes[1, 1].set_ylabel('Average Value')
            axes[1, 1].set_xticks(x + width * (len(video_names) - 1) / 2)
            axes[1, 1].set_xticklabels(metrics_names)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plots saved to: {output_path}")
        
        plt.show()


def main():
    """Command-line interface for metrics analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze fall detection metrics from JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to JSON metrics file")
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for plots and reports",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate summary report",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate visualization plots",
    )
    
    args = parser.parse_args()
    
    # Analyze metrics
    analyzer = MetricsAnalyzer()
    
    try:
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
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 