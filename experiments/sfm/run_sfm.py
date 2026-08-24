"""
JAX inference stage entrypoint: one long-running process per SLURM array task,
iterating its strided share of (stim_id, variant) rows. Pure JAX on the GPU (the
torch front-end ran in the preprocessing stage).

Usage:
  uv run python experiments/sfm/run_sfm.py --manifest M.tsv --config sfm_base \
      --task-id T --num-tasks N [--expect-warm] [--limit K] [--only-missing]
"""
import os

# Persistent compile cache + thread pinning must precede jax/numpy heavy use.
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
from experiments.sfm import sfm_config as cfg  # noqa: E402


def configure_jax_cache():
    d = cfg.assert_writable_path(cfg.JAX_CACHE_DIR)
    d.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(d))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


configure_jax_cache()

import numpy as np  # noqa: E402
from experiments.sfm import bundles, worklist  # noqa: E402
from experiments.sfm.algorithm import infer_window  # noqa: E402


def atomic_write(path: Path, writer):
    path = cfg.assert_writable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # suffix must end in .npz so np.savez does not append its own extension
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp" + path.suffix)
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="sfm_base")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=-1, help="-1 = config default")
    ap.add_argument("--out-root", default=str(cfg.RESULTS_DIR / "windows"))
    ap.add_argument("--expect-warm", action="store_true",
                    help="abort if the first window pays a long compile (cache miss)")
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--log-traces", action="store_true",
                    help="write thinned all-latent traces.npz per window "
                         "(logging-only: config name and content hash are unchanged)")
    ap.add_argument("--trace-thin", type=int, default=0, help="0 = config default")
    args = ap.parse_args()

    all_cfgs = dict(cfg.CONFIGS)
    all_cfgs.update(cfg.pilot_config_grid())
    mcfg = all_cfgs[args.config]
    if args.log_traces:
        import dataclasses as _dc
        mcfg = _dc.replace(mcfg, log_traces=True,
                           trace_thin=args.trace_thin or mcfg.trace_thin)

    rows = worklist.claim(worklist.read_manifest(args.manifest),
                          args.task_id, args.num_tasks)
    if args.limit:
        rows = rows[:args.limit]

    import dataclasses
    import subprocess
    git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True).stdout.strip()
    print(f"task {args.task_id}/{args.num_tasks}: {len(rows)} rows, config={mcfg.name} "
          f"(hash {mcfg.content_hash()}), git={git_rev}, manifest={args.manifest}, "
          f"out_root={args.out_root}, seed_arg={args.seed}, "
          f"devices={jax.devices()}", flush=True)
    print("CONFIG " + json.dumps(dataclasses.asdict(mcfg), sort_keys=True), flush=True)

    t_start = time.time()
    done = skipped = 0
    times = []
    for sid, variant in rows:
        if worklist.is_done(args.out_root, mcfg.name, variant, sid):
            skipped += 1
            continue
        bpath = bundles.bundle_path(cfg.BUNDLES_DIR, sid, variant)
        if not bpath.exists():
            print(f"MISSING bundle {bpath} — skipping", flush=True)
            continue
        arrays, meta = bundles.load_bundle(bpath)
        seed = mcfg.seed if args.seed < 0 else args.seed
        t0 = time.time()
        try:
            results, out_arrays, traces = infer_window(arrays, mcfg, seed=seed)
        except Exception as e:  # degenerate windows etc.: record + move on
            rdir = worklist.result_dir(args.out_root, mcfg.name, variant, sid)
            atomic_write(rdir / "results.json", lambda tmp: Path(tmp).write_text(
                json.dumps({"stim_id": sid, "variant": variant, "error": str(e),
                            "config": mcfg.name})))
            print(f"ERROR {sid}/{variant}: {e}", flush=True)
            done += 1
            continue
        dt = time.time() - t0
        times.append(dt)

        if args.expect_warm and done == 0 and dt > 600:
            print(f"ABORT: first window took {dt:.0f}s with --expect-warm "
                  f"(XLA cache miss?)", flush=True)
            sys.exit(3)

        results.update({"stim_id": sid, "variant": variant, "bundle_meta": meta,
                        "wall_s": dt})
        rdir = worklist.result_dir(args.out_root, mcfg.name, variant, sid)
        atomic_write(rdir / "assignments.npz",
                     lambda tmp: np.savez_compressed(tmp, **out_arrays))
        if traces:  # flat "phase.latent" keys, e.g. "f2_track.blob_means"
            flat = {f"{ph}.{k}": v for ph, d in traces.items() for k, v in d.items()}
            atomic_write(rdir / "traces.npz",
                         lambda tmp: np.savez_compressed(tmp, **flat))
        done += 1
        med = np.median(times)
        print(f"[{done}/{len(rows)}] stim={sid} obj={meta.get('object_id')} "
              f"tex={meta.get('texture_id')} vp={meta.get('viewpoint_id')} "
              f"size={meta.get('size_deg')} variant={variant} seed={seed} "
              f"roi_fb={results['roi_fallback']} gate={meta.get('evidence_gate', False)} "
              f"jacc={results['mean_roi_jaccard']:.3f} "
              f"acc={results['mean_probe_accuracy']:.3f} {dt:.1f}s "
              f"(median {med:.1f}s)", flush=True)
        atomic_write(rdir / "results.json",
                     lambda tmp: Path(tmp).write_text(json.dumps(results, indent=1)))

    print(f"task done: {done} inferred, {skipped} skipped, "
          f"median {np.median(times) if times else float('nan'):.1f}s/window, "
          f"total {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
