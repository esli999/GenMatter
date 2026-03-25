#!/usr/bin/env python3
"""CLI dispatcher for GenMatter experiments and postprocessing."""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()

# Value: script path relative to repo root, or (path, default argv prefix)
COMMANDS = {
    "populate-data": "scripts/populate_genmatter_data.py",
    "gestalt": "experiments/gestalt/run_gestalt.py",
    "gestalt-depth-ablation": "experiments/gestalt/run_gestalt_depth_ablation.py",
    "davis-extract-dino": "experiments/davis/dino_extractor.py",
    "davis-extract-sam-frame0": "experiments/davis/sam_frame0_extractor.py",
    "davis-extract-3d-motion": "preprocessing/motion_extraction_3d/run_davis_3d_motion.py",
    "download-tapvid-davis": "scripts/download_tapvid_davis.py",
    "davis-preprocess": "scripts/davis_preprocess.py",
    # DAVIS: 3 experiment types × SAM on/off (six runners). Legacy names = SAM (same as ``*-sam``).
    "davis-tracking-sam": (
        "experiments/davis/run_davis_tracking.py",
        ["--use-sam"],
    ),
    "davis-tracking-no-sam": (
        "experiments/davis/run_davis_tracking.py",
        ["--no-use-sam"],
    ),
    "davis-tracking": (
        "experiments/davis/run_davis_tracking.py",
        ["--use-sam"],
    ),
    "davis-subsampling-sam": (
        "experiments/davis/run_davis_subsampling.py",
        ["--use-sam"],
    ),
    "davis-subsampling-no-sam": (
        "experiments/davis/run_davis_subsampling.py",
        ["--no-use-sam"],
    ),
    "davis-subsampling": (
        "experiments/davis/run_davis_subsampling.py",
        ["--use-sam"],
    ),
    "davis-ablation-sam": (
        "experiments/davis/run_davis_ablation.py",
        ["--use-sam"],
    ),
    "davis-ablation-no-sam": (
        "experiments/davis/run_davis_ablation.py",
        ["--no-use-sam"],
    ),
    "davis-ablation": (
        "experiments/davis/run_davis_ablation.py",
        ["--use-sam"],
    ),
    "davis-ablation-2-sam": (
        "experiments/davis/run_davis_ablation_2.py",
        ["--use-sam"],
    ),
    "davis-ablation-2-no-sam": (
        "experiments/davis/run_davis_ablation_2.py",
        ["--no-use-sam"],
    ),
    "davis-ablation-2": (
        "experiments/davis/run_davis_ablation_2.py",
        ["--use-sam"],
    ),
    "davis-ablation-3-sam": (
        "experiments/davis/run_davis_ablation_3.py",
        ["--use-sam"],
    ),
    "davis-ablation-3-no-sam": (
        "experiments/davis/run_davis_ablation_3.py",
        ["--no-use-sam"],
    ),
    "davis-ablation-3": (
        "experiments/davis/run_davis_ablation_3.py",
        ["--use-sam"],
    ),
    "davis-ablation-4-sam": (
        "experiments/davis/run_davis_ablation_4.py",
        ["--use-sam"],
    ),
    "davis-ablation-4-no-sam": (
        "experiments/davis/run_davis_ablation_4.py",
        ["--no-use-sam"],
    ),
    "davis-ablation-4": (
        "experiments/davis/run_davis_ablation_4.py",
        ["--use-sam"],
    ),
    "cotracker": "experiments/baselines/run_cotracker.py",
    "psychophysics-benchmark": (
        "experiments/psychophysics/run_psychophysics_benchmark.py",
        ["--mode", "full"],
    ),
    "psychophysics-rdk-ablation-fixed": (
        "experiments/psychophysics/run_psychophysics_benchmark.py",
        ["--mode", "rdk-ablation-fixed"],
    ),
    "psychophysics-rdk-ablation-adaptive": (
        "experiments/psychophysics/run_psychophysics_benchmark.py",
        ["--mode", "rdk-ablation-adaptive"],
    ),
    "rdk-preprocess": "preprocessing/random_dot_kinematograms/run_rdk_preprocess.py",
    "postprocess-psychophysics": "postprocessing/postprocess_psychophysics_correlation.py",
    "postprocess-gestalt": "postprocessing/postprocess_gestalt.py",
    "postprocess-gestalt-ablation": "postprocessing/postprocess_gestalt.py",
    "postprocess-davis": "postprocessing/postprocess_davis.py",
}


def _command_script_and_defaults(spec):
    if isinstance(spec, tuple):
        return spec[0], list(spec[1])
    return spec, []


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

    rel_path, default_args = _command_script_and_defaults(COMMANDS[args.command])
    script = REPO_ROOT / rel_path
    if not script.exists():
        print(f"Error: script {script} not found", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(script)] + default_args + args.extra
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
