#!/usr/bin/env python3
"""RDK psychophysics benchmark: full model, rdk-ablation-fixed, or rdk-ablation-adaptive (RANSAC motion only)."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import jax
import numpy as np
from jax.random import key as jkey
from scipy.stats import pearsonr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import config

from genmatter.psychophysics_utils import (
    model_prediction_on_stimulus,
    model_prediction_on_stimulus_ablation1,
    model_prediction_on_stimulus_ablation2,
)

# JAX compilation cache (same pattern as DAVIS experiments)
_cache_dir = os.path.join(str(REPO_ROOT), ".jax_cache")
os.makedirs(_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.experimental.compilation_cache.compilation_cache.set_cache_dir(_cache_dir)

HUMAN_RESULTS = {
    "s1_c1": 86.0,
    "s1_c2": 92.0,
    "s1_c4": 64.0,
    "s1_c5": 82.0,
    "s1_c6": 84.0,
    "s1_c7": 12.0,
    "s1_c8": 90.0,
    "s1_c9": 14.0,
    "s1_c10": 54.0,
    "s2_c1": 42.0,
    "s2_c2": 86.0,
    "s2_c4": 80.0,
    "s2_c5": 78.0,
    "s2_c6": 18.0,
    "s2_c7": 38.0,
    "s2_c8": 66.0,
    "s2_c9": 30.0,
    "s2_c10": 88.0,
    "s3_c1": 84.0,
    "s3_c2": 18.0,
    "s3_c4": 94.0,
    "s3_c5": 90.0,
    "s3_c6": 26.0,
    "s3_c7": 44.0,
    "s3_c8": 90.0,
    "s3_c9": 36.0,
    "s3_c10": 84.0,
}

MODE_TO_PREDICT_FN = {
    "full": model_prediction_on_stimulus,
    "rdk-ablation-fixed": model_prediction_on_stimulus_ablation1,
    "rdk-ablation-adaptive": model_prediction_on_stimulus_ablation2,
}


def _is_familiarization(stim_id: str) -> bool:
    """Exclude e.g. ``fam_1``, ``fam_2`` from the benchmark (not counted, not run)."""
    return stim_id.startswith("fam_")


def _gt_as_int(gt):
    if isinstance(gt, bool):
        return 1 if gt else 0
    return int(gt)


def _resolve_repro_keys_path(explicit: str | None) -> Path | None:
    """Prefer explicit path, then ``GENMATTER_RDK_REPRO_KEYS_PATH`` / ``reproducibility_keys.json``, then legacy ``keys.json``."""
    if explicit:
        p = Path(explicit).resolve()
        return p if p.is_file() else None
    primary = config.rdk_reproducibility_keys_json_path()
    if primary.is_file():
        return primary
    legacy = config.RDK_ROOT / "keys.json"
    if legacy.is_file():
        return legacy
    return None


def _load_reproducibility_keys(path: Path) -> dict[str, int]:
    with open(path, "r") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


def main():
    parser = argparse.ArgumentParser(description="RDK psychophysics benchmark (GenMatter)")
    parser.add_argument(
        "--mode",
        choices=list(MODE_TO_PREDICT_FN.keys()),
        required=True,
        help="full | rdk-ablation-fixed | rdk-ablation-adaptive",
    )
    parser.add_argument("--outer-trials", type=int, default=10, help="Outer random trials per stimulus")
    parser.add_argument("--num-runs", type=int, default=5, help="Inner parallel chains (num_runs)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: config.PSYCHOPHYSICS_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--no-repro-keys",
        action="store_true",
        help="Ignore reproducibility_keys.json (use random init_key_int per stimulus)",
    )
    parser.add_argument(
        "--repro-keys-path",
        type=str,
        default=None,
        help="Path to JSON map stim_id -> integer seed (overrides default location)",
    )
    parser.add_argument(
        "--require-repro-keys",
        action="store_true",
        help="Fail if the keys file is missing or a benchmark stim_id is not in the map",
    )
    args = parser.parse_args()

    if args.repro_keys_path:
        rp = Path(args.repro_keys_path).resolve()
        if not rp.is_file():
            print(f"Error: --repro-keys-path not found: {rp}", file=sys.stderr)
            sys.exit(1)

    cfg_path = config.rdk_configs_json_path()
    if not cfg_path.is_file():
        print(f"Error: missing {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path, "r") as f:
        stimuli_configs = json.load(f)

    repro_keys_path: Path | None = None
    repro_map: dict[str, int] | None = None
    if not args.no_repro_keys:
        repro_keys_path = _resolve_repro_keys_path(args.repro_keys_path)
        if repro_keys_path is not None:
            repro_map = _load_reproducibility_keys(repro_keys_path)
        elif args.require_repro_keys:
            tried = config.rdk_reproducibility_keys_json_path()
            print(
                f"Error: no reproducibility keys file found (tried {tried} and {config.RDK_ROOT / 'keys.json'})",
                file=sys.stderr,
            )
            sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else config.PSYCHOPHYSICS_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    predict_fn = MODE_TO_PREDICT_FN[args.mode]
    n_total_json = len(stimuli_configs)
    n_benchmark = sum(1 for sid in stimuli_configs if not _is_familiarization(sid))
    n_fam = n_total_json - n_benchmark

    outer = args.outer_trials
    inner = args.num_runs
    samples_per_stim = outer * inner

    print("=" * 70)
    print("Psychophysics benchmark — configuration")
    print("=" * 70)
    print(f"  Mode:              {args.mode}")
    print(f"  RDK root:          {config.RDK_ROOT}")
    print(f"  Config JSON:       {cfg_path}")
    if repro_keys_path is not None:
        print(f"  Repro keys:        {len(repro_map)} entries from {repro_keys_path}")
    else:
        print("  Repro keys:        (none) random init_key_int per stimulus")
    print(f"  Motion:            RANSAC")
    print(f"  Stimuli to run:    {n_benchmark} (skip {n_fam} fam_*; {n_total_json} keys in JSON)")
    print("  Per stimulus:")
    print(f"    Outer trials:    {outer}  (new RNG split each time; independent predict_fn calls)")
    print(f"    Inner num_runs:  {inner}  (parallel Gibbs chains per outer trial, vmapped)")
    print(
        f"    Reported %%:      mean over {samples_per_stim} binary outcomes "
        f"({outer}×{inner} chain ends), then ×100"
    )
    print(
        f"  Work (approx):     {n_benchmark} stimuli × {outer} outer × 1 predict_fn "
        f"(each runs {inner} chains in parallel)"
    )
    print("=" * 70)

    model_results = {}
    count = 0
    warned_missing_repro_key = False

    for stim_id, stimulus in stimuli_configs.items():
        if _is_familiarization(stim_id):
            continue
        count += 1
        print(f"RUNNING STIMULUS {count}/{n_benchmark} ({stim_id})")

        config_num = stimulus["config_num"]
        start_frame = stimulus["start_frame"]
        end_frame = stimulus["end_frame"]
        probe_timestep = stimulus["probe_timestep"]

        if random.random() < 0.5:
            red_point = stimulus["point_1"]
            green_point = stimulus["point_2"]
        else:
            red_point = stimulus["point_2"]
            green_point = stimulus["point_1"]

        ground_truth = stimulus["ground_truth"]
        npz_path = config.rdk_npz_path(config_num)
        if not npz_path.is_file():
            print(f"Error: missing {npz_path}", file=sys.stderr)
            sys.exit(1)

        if repro_map is not None and stim_id in repro_map:
            init_key_int = int(repro_map[stim_id])
        else:
            if repro_map is not None and args.require_repro_keys:
                print(
                    f"Error: stim_id {stim_id!r} missing from reproducibility keys",
                    file=sys.stderr,
                )
                sys.exit(1)
            if repro_map is not None and stim_id not in repro_map and not warned_missing_repro_key:
                print(
                    "Warning: some stim_ids are missing from the reproducibility map; "
                    f"using random init_key_int for those (first missing: {stim_id!r}). "
                    "Use --require-repro-keys to fail instead.",
                    file=sys.stderr,
                )
                warned_missing_repro_key = True
            init_key_int = np.random.randint(1, int(1e9))
        key = jkey(init_key_int)
        trial_outputs = []
        for _ in tqdm(range(args.outer_trials), leave=False):
            key, exp_key = jax.random.split(key)
            result = predict_fn(
                exp_key,
                npz_path,
                start_frame,
                end_frame,
                probe_timestep,
                red_point,
                green_point,
                num_runs=args.num_runs,
            )
            trial_outputs.append(result)

        stacked = np.mean(np.concatenate([np.asarray(t) for t in trial_outputs])) * 100.0
        model_results[stim_id] = float(stacked)
        print(f"stim_id: {stim_id}, Model Prediction: {stacked:.2f}%, ground_truth: {ground_truth}")

    model_vals = []
    human_vals = []
    stimuli_out = {}

    print("\nResults Summary:")
    print("=" * 70)
    print(f"{'Stimulus ID':12} {'Model Prediction':>15} {'Human Results':>15} {'Ground Truth':>15}")
    print("-" * 70)

    for stim_id in sorted(model_results.keys()):
        gt_raw = stimuli_configs[stim_id]["ground_truth"]
        gt_int = _gt_as_int(gt_raw)
        stimuli_out[stim_id] = {
            "model_percent": model_results[stim_id],
            "ground_truth": gt_raw,
        }
        if stim_id not in HUMAN_RESULTS:
            continue
        model_pred = model_results[stim_id]
        human_res = HUMAN_RESULTS[stim_id]
        model_vals.append(model_pred)
        human_vals.append(human_res)
        print(f"{stim_id:12} {model_pred:15.2f} {human_res:15.2f} {gt_int:15}")

    print("-" * 70)

    model_vals = np.array(model_vals, dtype=np.float64)
    human_vals = np.array(human_vals, dtype=np.float64)
    r_pearson, p_value = pearsonr(model_vals, human_vals)
    r_squared = float(r_pearson**2)

    print(f"\nPearson r: {r_pearson:.4f} (p={p_value:.4g})")
    print(f"R-squared: {r_squared:.3f}")

    safe_name = args.mode.replace("-", "_")
    json_path = out_dir / f"{safe_name}_benchmark.json"
    payload = {
        "mode": args.mode,
        "motion": "ransac",
        "outer_trials": args.outer_trials,
        "num_runs": args.num_runs,
        "rdk_root": str(config.RDK_ROOT),
        "reproducibility_keys_path": str(repro_keys_path) if repro_keys_path else None,
        "stimuli": stimuli_out,
        "human_results": HUMAN_RESULTS,
        "metrics": {
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_value),
            "r_squared": r_squared,
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
