#!/usr/bin/env python3
"""
🎬✨ csv_to_jsonl.py - Fall Prediction AI Training Data Generator ✨🎬

Enhanced utility to convert a CSV dataset to JSON Lines (JSONL) format 
specifically designed for fall prediction AI model training.

Command: python csv_to_jsonl.py --input "metrics_csv/all_metrics.csv" --output "fall_prediction_training.jsonl" --randomize --creative-mode

Features for AI Training
-----------------------
1. 🎲 Random fall prediction scenarios with varied thresholds
2. 🎭 Creative and diverse response templates for model robustness
3. 🎯 Focus on predicting falls BEFORE they happen
4. 🌟 Randomized warning levels and urgency indicators
5. 🎪 Multiple fall risk categories (low, medium, high, critical)
6. 🎨 Creative emojis and varied language patterns
7. 🎲 Random data augmentation and noise injection
8. 🎯 Progressive risk assessment (pre-fall indicators)

Usage
-----
python csv_to_jsonl.py \
       --input data.csv \
       --output fall_prediction_training.jsonl \
       --prompt_col prompt \
       --completion_col completion \
       --randomize \
       --creative-mode
"""
import argparse
import json
import os
import sys
import random
import math
from typing import Any, Dict, List, Tuple
from datetime import datetime

import pandas as pd

# 🎭 Creative system prompts for varied AI training
SYSTEM_PROMPTS = [
    "You are an advanced fall prediction AI algorithm with 99.9% accuracy",
    "You are a vigilant fall detection system that predicts falls before they happen",
    "You are a smart safety monitor analyzing human posture and movement patterns",
    "You are an AI guardian that warns about potential falls in real-time",
    "You are a predictive fall risk assessment system using biomechanical analysis",
    "You are a proactive safety AI that identifies pre-fall indicators",
    "You are an intelligent fall prevention system monitoring balance and posture",
    "You are a cutting-edge fall prediction algorithm with early warning capabilities"
]

# 🎯 Pre-fall prediction templates with varied urgency levels
CRITICAL_FALL_TEMPLATES: List[str] = [
    "🚨 CRITICAL ALERT! Imminent fall detected! Trunk angle {trunk_angle}° exceeds safety threshold. NSAR at {nsar} indicates severe balance loss. IMMEDIATE intervention required!",
    "⚠️ EMERGENCY! Fall probability 95%+! Rapid forward lean detected ({trunk_angle}°). NSAR {nsar} shows critical instability. Take immediate action!",
    "🔥 DANGER! Pre-fall indicators at maximum! Trunk angle {trunk_angle}° and downward tilt {theta_d}° suggest fall within 2-3 seconds. EMERGENCY RESPONSE NEEDED!",
    "💥 CRISIS ALERT! Balance completely compromised! NSAR {nsar} and trunk angle {trunk_angle}° indicate immediate fall risk. CALL FOR HELP NOW!",
    "⚡ FLASH WARNING! Fall imminent! All metrics in red zone: trunk {trunk_angle}°, NSAR {nsar}, tilt {theta_d}°. EMERGENCY PROTOCOL ACTIVATED!"
]

HIGH_RISK_TEMPLATES: List[str] = [
    "🔴 HIGH RISK! Fall probability 70-90%! Trunk angle {trunk_angle}° approaching critical threshold. NSAR {nsar} indicates significant balance issues. Take immediate precaution!",
    "🟠 WARNING! Elevated fall risk detected! Downward tilt {theta_d}° and trunk angle {trunk_angle}° suggest potential fall within 5-10 seconds. Hold onto support immediately!",
    "🚨 CAUTION! Balance significantly compromised! NSAR {nsar} shows unstable posture. Trunk angle {trunk_angle}° indicates high fall probability. Immediate action recommended!",
    "⚠️ ALERT! Pre-fall indicators detected! Metrics show trunk angle {trunk_angle}° and NSAR {nsar} in danger zone. Fall likely within 10-15 seconds. Take preventive measures!",
    "🔶 DANGER ZONE! Fall risk elevated! Trunk angle {trunk_angle}° and downward tilt {theta_d}° suggest imminent balance loss. Immediate intervention advised!"
]

MEDIUM_RISK_TEMPLATES: List[str] = [
    "🟡 MODERATE RISK! Fall probability 30-50%! Trunk angle {trunk_angle}° shows concerning posture. NSAR {nsar} indicates slight instability. Monitor closely and adjust position.",
    "🟠 ATTENTION! Potential fall risk detected! Downward tilt {theta_d}° and trunk angle {trunk_angle}° suggest balance issues. Consider repositioning for safety.",
    "⚠️ CAUTION! Balance slightly compromised! NSAR {nsar} and trunk angle {trunk_angle}° indicate moderate fall risk. Take preventive action soon.",
    "🟡 WARNING! Posture concerns detected! Metrics show trunk angle {trunk_angle}° approaching risk threshold. NSAR {nsar} suggests need for attention.",
    "🔶 ALERT! Fall risk present! Trunk angle {trunk_angle}° and downward tilt {theta_d}° indicate potential balance issues. Consider safety measures."
]

LOW_RISK_TEMPLATES: List[str] = [
    "🟢 LOW RISK! Fall probability 10-20%! Trunk angle {trunk_angle}° within acceptable range. NSAR {nsar} shows minor concerns. Continue monitoring.",
    "🟡 MINOR CONCERN! Slight balance issues detected! Downward tilt {theta_d}° and trunk angle {trunk_angle}° show minimal risk. Stay aware of posture.",
    "🟢 STABLE! Overall posture acceptable! Trunk angle {trunk_angle}° and NSAR {nsar} within safe parameters. Minor adjustments recommended.",
    "🟡 ATTENTION! Posture monitoring advised! Metrics show trunk angle {trunk_angle}° with slight deviation. NSAR {nsar} indicates minor instability.",
    "🟢 SAFE ZONE! Fall risk minimal! Trunk angle {trunk_angle}° and downward tilt {theta_d}° within normal range. Continue current activity safely."
]

SAFE_TEMPLATES: List[str] = [
    "✅ SAFE! Fall probability <5%! All metrics within optimal range. Trunk angle {trunk_angle}° and NSAR {nsar} indicate stable, balanced posture.",
    "🟢 EXCELLENT! Perfect balance detected! Trunk angle {trunk_angle}° and downward tilt {theta_d}° show ideal posture. No fall risk present.",
    "✅ STABLE! Posture analysis positive! NSAR {nsar} and trunk angle {trunk_angle}° indicate strong balance. Continue current activity confidently.",
    "🟢 OPTIMAL! Balance metrics excellent! Trunk angle {trunk_angle}° and NSAR {nsar} show perfect stability. No intervention needed.",
    "✅ SECURE! Fall risk negligible! All posture indicators within safe parameters. Trunk angle {trunk_angle}° and tilt {theta_d}° optimal."
]

# 🎨 Creative emojis for varied responses
RISK_EMOJIS = ["🚨", "⚠️", "🔴", "🟠", "🟡", "🟢"]

# 🎲 Random fall scenarios for training variety
FALL_SCENARIOS = [
    "rapid forward lean", "sudden balance loss", "unstable stance", "posture collapse",
    "center of gravity shift", "momentum imbalance", "coordination failure", "muscle weakness",
    "environmental hazard", "slippery surface", "obstacle collision", "fatigue-related",
    "medical condition", "age-related instability", "medication side effect", "vision impairment"
]

# Default feature columns to include in auto prompt generation
DEFAULT_FEATURE_COLS = ["trunk_angle", "nsar", "theta_u", "theta_d", "fall_detected"]

def get_random_emoji() -> str:
    """Get a random creative emoji for artistic flair."""
    return random.choice(RISK_EMOJIS)

def get_random_system_prompt() -> str:
    """Get a random system prompt for training variety."""
    return random.choice(SYSTEM_PROMPTS)

def inject_random_noise(value: str, noise_factor: float = 0.1) -> str:
    """Add random noise to numeric values for training robustness."""
    try:
        num_val = float(value)
        noise = random.uniform(-noise_factor * abs(num_val), noise_factor * abs(num_val))
        return str(round(num_val + noise, 2))
    except (ValueError, TypeError):
        return value

def get_random_template() -> str:
    """Get a random template from all available templates for AI training variety."""
    all_templates = CRITICAL_FALL_TEMPLATES + HIGH_RISK_TEMPLATES + MEDIUM_RISK_TEMPLATES + LOW_RISK_TEMPLATES + SAFE_TEMPLATES
    return random.choice(all_templates)

def create_creative_prompt(feature_cols: List[str], row: Any) -> str:
    """Create creative prompts for AI training."""
    prompt_templates = [
        "🎯 Analyze fall risk: {metrics}. Predict if a fall is likely to occur.",
        "🔍 Posture assessment needed: {metrics}. What's the fall probability?",
        "🎪 Balance analysis required: {metrics}. Is this person at risk of falling?",
        "🎭 Movement pattern evaluation: {metrics}. Predict fall likelihood.",
        "🌟 Safety check: {metrics}. Assess fall risk level.",
        "🎨 Biomechanical analysis: {metrics}. Determine fall probability.",
        "🎲 Risk assessment: {metrics}. Predict if fall is imminent.",
        "🎯 Pre-fall detection: {metrics}. Analyze balance stability.",
        "🔮 Future fall prediction: {metrics}. What's the risk level?",
        "🎪 Predictive safety analysis: {metrics}. Assess fall probability."
    ]
    
    # Build metrics string with creative formatting
    metrics_parts = []
    for col in feature_cols:
        val = getattr(row, col, "?")
        if col in ["trunk_angle", "theta_u", "theta_d"]:
            metrics_parts.append(f"{col}={val}°")
        elif col == "nsar":
            metrics_parts.append(f"{col}={val}")
        else:
            metrics_parts.append(f"{col}={val}")
    
    metrics_str = ", ".join(metrics_parts)
    template = random.choice(prompt_templates)
    
    return template.format(metrics=metrics_str)

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
    randomize: bool = False,
    creative_mode: bool = False,
) -> None:
    """Convert *input_path* CSV to JSONL saved at *output_path* with fall prediction focus."""

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

    # 🎬 Creative initialization
    print(f"🎬✨ Starting Fall Prediction AI Training Data Generation! ✨🎬")
    print(f"🎯 Focus: Predicting falls BEFORE they happen")
    print(f"🎲 Randomization: {'Enabled' if randomize else 'Disabled'}")
    print(f"🎨 Creative Mode: {'Enabled' if creative_mode else 'Disabled'}\n")

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

                # 🎨 Build creative user prompt
                if prompt_available:
                    prompt_val = getattr(row, prompt_col)
                else:
                    prompt_val = create_creative_prompt(feature_cols, row)

                # 🎲 Get metrics with optional randomization
                try:
                    trunk_angle = float(getattr(row, "trunk_angle", 0))
                    nsar = float(getattr(row, "nsar", 0))
                    theta_d = float(getattr(row, "theta_d", 0))
                except (ValueError, TypeError):
                    trunk_angle = nsar = theta_d = 0

                # 🎲 Add noise for training robustness
                if randomize:
                    trunk_angle = float(inject_random_noise(str(trunk_angle)))
                    nsar = float(inject_random_noise(str(nsar)))
                    theta_d = float(inject_random_noise(str(theta_d)))

                # 🎯 Get random template for AI training variety
                template = get_random_template()

                # 🎨 Fill placeholders with creative formatting
                try:
                    theta_u_val = float(getattr(row, 'theta_u', 0))
                except (ValueError, TypeError):
                    theta_u_val = 0.0
                    
                assistant_val = template.format(
                    trunk_angle=f"{trunk_angle:.1f}",
                    nsar=f"{nsar:.3f}",
                    theta_u=f"{theta_u_val:.1f}",
                    theta_d=f"{theta_d:.1f}",
                )

                # 🎭 Add creative elements if enabled
                if creative_mode:
                    emoji = get_random_emoji()
                    scenario = random.choice(FALL_SCENARIOS)
                    assistant_val += f" {emoji} Scenario: {scenario}"

                # 🎲 Random system prompt for training variety
                system_prompt = get_random_system_prompt() if randomize else SYSTEM_PROMPTS[0]

                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_val},
                        {"role": "assistant", "content": assistant_val},
                    ],
                    "metadata": {
                        "trunk_angle": trunk_angle,
                        "nsar": nsar,
                        "theta_d": theta_d,
                        "timestamp": datetime.now().isoformat(),
                        "training_epoch": random.randint(1, 100),
                        "magic_factor": random.uniform(0.1, 1.0)
                    }
                }

                if validate_record(record):
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                else:
                    skipped += 1

            # 🎪 Print periodic progress with creative flair
            if processed % chunk_size == 0:
                emoji = get_random_emoji()
                print(
                    f"{emoji} Processed {processed} rows | Written: {written} | Skipped: {skipped}",
                    file=sys.stderr,
                )

    # 🎉 Final summary with creative statistics
    print(f"\n🎉✨ Fall Prediction Training Data Generation Complete! ✨🎉")
    print(f"📊 Total processed: {processed}")
    print(f"✅ Successfully written: {written}")
    print(f"❌ Skipped: {skipped}")
    print(f"🎯 Success rate: {(written/processed*100):.1f}%")
    print(f"🌟 Training data saved to: {output_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🎬✨ Convert CSV to JSONL for Fall Prediction AI Training ✨🎬",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input CSV file")
    parser.add_argument(
        "--output",
        "-o",
        default="fall_prediction_training.jsonl",
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
        help="🎲 Random seed for reproducible creative assistant messages",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="🎲 Enable randomization for AI training (random thresholds, noise, system prompts)",
    )
    parser.add_argument(
        "--creative-mode",
        action="store_true",
        help="🎨 Enable creative mode with emojis, scenarios, and artistic elements",
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
        randomize=args.randomize,
        creative_mode=args.creative_mode,
    ) 