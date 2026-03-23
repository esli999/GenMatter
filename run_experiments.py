#!/usr/bin/env python3
"""CLI dispatcher for GenMatter experiments and postprocessing."""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()

COMMANDS = {
    "populate-data": "scripts/populate_genmatter_data.py",
    "gestalt": "experiments/gestalt/run_gestalt.py",
    "gestalt-depth-ablation": "experiments/gestalt/run_gestalt_depth_ablation.py",
    "davis-extract-dino": "experiments/davis/dino_extractor.py",
    "davis-tracking": "experiments/davis/run_davis_tracking.py",
    "davis-subsampling": "experiments/davis/run_davis_subsampling.py",
    "davis-ablation": "experiments/davis/run_davis_ablation.py",
    "cotracker": "experiments/baselines/run_cotracker.py",
    "postprocess-gestalt": "postprocessing/postprocess_gestalt.py",
    "postprocess-gestalt-ablation": "postprocessing/postprocess_gestalt_ablation.py",
    "postprocess-davis": "postprocessing/postprocess_davis.py",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run GenMatter experiments or postprocessing steps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available commands:\n" + "\n".join(f"  {k}" for k in COMMANDS),
    )
    parser.add_argument(
        "command",
        choices=COMMANDS.keys(),
        help="Experiment, data setup, or postprocessing step to run",
    )
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments forwarded to the script")
    args = parser.parse_args()

    script = REPO_ROOT / COMMANDS[args.command]
    if not script.exists():
        print(f"Error: script {script} not found", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(script)] + args.extra
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
