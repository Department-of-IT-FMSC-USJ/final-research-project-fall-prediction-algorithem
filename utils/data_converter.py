"""Data conversion utilities for fall detection datasets."""

import argparse
import json
import os
import sys
import random
from typing import Any, Dict, List, Optional
import pandas as pd


class DataConverter:
    """Handles data format conversions for fall detection datasets."""
    
    # Message templates for fall detection
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
    
    # Default feature columns
    DEFAULT_FEATURE_COLS = ["trunk_angle", "nsar", "theta_u", "theta_d"]
    
    def __init__(self, system_prompt: str = "You are a fall detection algorithm", seed: Optional[int] = None):
        """Initialize data converter.
        
        Args:
            system_prompt: System prompt for AI training
            seed: Random seed for reproducible results
        """
        self.system_prompt = system_prompt
        if seed is not None:
            random.seed(seed)
    
    def csv_to_jsonl(self, input_path: str, output_path: str, 
                    prompt_col: str = "prompt", completion_col: str = "completion",
                    chunk_size: int = 10_000, feature_cols: Optional[List[str]] = None) -> Dict[str, int]:
        """Convert CSV file to JSONL format for AI training.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output JSONL file
            prompt_col: Column name for prompts
            completion_col: Column name for completions (for compatibility)
            chunk_size: Number of rows to process at once
            feature_cols: Feature columns to include in auto-generated prompts
            
        Returns:
            Dictionary with processing statistics
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Validate required columns exist
        try:
            header_cols = list(pd.read_csv(input_path, nrows=0).columns)
        except Exception as err:
            raise ValueError(f"Failed to read CSV header: {err}")
        
        prompt_available = prompt_col in header_cols
        
        # Use provided feature columns or defaults that exist in file
        if feature_cols is None:
            feature_cols = [col for col in self.DEFAULT_FEATURE_COLS if col in header_cols]
        
        if not prompt_available and not feature_cols:
            raise ValueError(
                "No prompt column available and none of the default feature columns are present to generate a prompt."
            )
        
        stats = {"processed": 0, "written": 0, "skipped": 0}
        
        # Process in chunks
        with open(output_path, "w", encoding="utf-8") as fout:
            try:
                reader = pd.read_csv(input_path, chunksize=chunk_size, dtype=str)
            except Exception as err:
                raise ValueError(f"Failed to read CSV: {err}")
            
            for chunk_idx, chunk in enumerate(reader, start=1):
                for row in chunk.itertuples(index=False):
                    stats["processed"] += 1
                    
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
                    
                    template_source = self.FALL_TEMPLATES if fall_bool else self.SAFE_TEMPLATES
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
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt_val},
                            {"role": "assistant", "content": assistant_val},
                        ]
                    }
                    
                    if self._validate_record(record):
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stats["written"] += 1
                    else:
                        stats["skipped"] += 1
                
                # Print periodic progress
                if stats["processed"] % chunk_size == 0:
                    print(
                        f"Processed {stats['processed']} rows | Written: {stats['written']} | Skipped: {stats['skipped']}",
                        file=sys.stderr,
                    )
        
        return stats
    
    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate that record is JSON serializable and contains non-empty user content.
        
        Args:
            record: Record to validate
            
        Returns:
            True if valid, False otherwise
        """
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
    
    def combine_json_metrics(self, input_dir: str, output_file: str) -> int:
        """Combine multiple JSON metrics files into a single file.
        
        Args:
            input_dir: Directory containing JSON metrics files
            output_file: Output combined file path
            
        Returns:
            Number of files combined
        """
        import glob
        
        # Find all JSON files in the directory
        json_files = glob.glob(os.path.join(input_dir, "*_metrics.json"))
        
        if not json_files:
            print(f"No JSON metrics files found in {input_dir}")
            return 0
        
        combined_data = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        combined_data.extend(data)
                    else:
                        combined_data.append(data)
                print(f"Added: {json_file}")
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
        
        # Save combined data
        with open(output_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        print(f"Combined {len(json_files)} files into {output_file}")
        return len(json_files)


def main():
    """Command-line interface for data conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a CSV file to JSONL with system/user/assistant keys.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input CSV file")
    parser.add_argument(
        "--output",
        "-o",
        default="metrics.jsonl",
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
        help="Number of rows to process at once",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible results",
    )
    
    args = parser.parse_args()
    
    # Convert CSV to JSONL
    converter = DataConverter(seed=args.seed)
    try:
        stats = converter.csv_to_jsonl(
            args.input,
            args.output,
            args.prompt_col,
            args.completion_col,
            args.chunk_size
        )
        
        print(
            f"✅ Conversion complete. Total processed: {stats['processed']}, "
            f"Written: {stats['written']}, Skipped: {stats['skipped']}"
        )
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 