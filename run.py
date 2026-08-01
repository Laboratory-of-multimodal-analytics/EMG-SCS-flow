#!/usr/bin/env python3
"""
Run the EMG pipeline from the command line.

Usage (from project root):
  python run.py <input_file> [--startstop] [--output-dir DIR]

Input: EDF, FIF, MAT, or a text "curves" export (.txt, pre-cut epochs).

Results go NEXT TO the recording by default, in a folder named "<subject> <file>"
where the subject is the name of the folder the recording sits in:

  (Zh14)/Th11-12.txt  ->  (Zh14)/(Zh14) Th11-12/

Half the subjects have a "Th11-12.txt", so the prefix is what keeps their result
folders apart once opened side by side or copied out. Nothing needs to be passed
for this -- pass --output-dir only to deliberately put results somewhere else,
which also gives up the prefix and the run-local template bank.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EMG pipeline for SCS / start-stop protocols. Supports EDF, FIF, MAT and text curves exports.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the recording (EDF, FIF, MAT, or a .txt curves export).",
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
        help="Put results here instead of next to the recording. "
             "Skips the subject prefix; templates drawn for this recording are "
             "not picked up from the default location either.",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Error: input file not found: {args.input_path}", file=sys.stderr)
        sys.exit(1)

    output_root = run_pipeline(
        args.input_path,
        output_dir=args.output_dir,
        startstop_mode=args.startstop,
    )
    print(f"Outputs saved under: {output_root}")


if __name__ == "__main__":
    main()
