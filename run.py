#!/usr/bin/env python3
"""
Run the EMG pipeline from the command line.

Usage (from project root):
  python run.py <input_file> [--startstop | --no-startstop] [--output-dir DIR] [--template-mode MODE]

Input: EDF, FIF, or MAT file. Outputs go to results/ or --output-dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emg_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EMG pipeline for SCS / start-stop protocols. Supports EDF, FIF, and MAT.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the recording (EDF, FIF, or MAT).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--startstop",
        action="store_true",
        dest="startstop",
        help="Run in StartStop mode (template matching on start/stop segments).",
    )
    mode.add_argument(
        "--no-startstop",
        action="store_false",
        dest="startstop",
        help="Run in stimulation-induced mode (config/amplitude crops, stimulus-centered epochs).",
    )
    parser.set_defaults(startstop=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory root (default: results/<filename_stem>).",
    )
    parser.add_argument(
        "--template-mode",
        choices=["per_file_mean", "config_grand_avg", "max_amp_only"],
        default="per_file_mean",
        help="Template mode for stimulation-induced detection (default: per_file_mean).",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Error: input file not found: {args.input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    startstop_mode = args.startstop

    output_root = run_pipeline(
        args.input_path,
        output_dir=output_dir,
        template_mode=args.template_mode,
        startstop_mode=startstop_mode,
    )
    print(f"Outputs saved under: {output_root}")


if __name__ == "__main__":
    main()
