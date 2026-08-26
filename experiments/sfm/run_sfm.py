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
    ap.add_argument("--memory", default="off",
                    help="MemoryConfig name from memory_config_grid(); results are "
                         "routed under <config>+<memory> when not 'off'")
    ap.add_argument("--batch-windows", type=int, default=1,
                    help=">1 = run B windows through one vmapped program "
                         "(Workstream 6; ~2x throughput on latency-bound phases)")
    args = ap.parse_args()

    all_cfgs = dict(cfg.CONFIGS)
    all_cfgs.update(cfg.pilot_config_grid())
    mcfg = all_cfgs[args.config]
    if args.log_traces:
        import dataclasses as _dc
        mcfg = _dc.replace(mcfg, log_traces=True,
                           trace_thin=args.trace_thin or mcfg.trace_thin)

    from experiments.sfm.memory_config import memory_config_grid
    mem = memory_config_grid()[args.memory]
    out_name = mem.result_name(mcfg.name)
    if not mem.is_off():
        from experiments.sfm.memory_algorithm import infer_window_mem

    rows = worklist.claim(worklist.read_manifest(args.manifest),
                          args.task_id, args.num_tasks)
    if args.limit:
        rows = rows[:args.limit]

    import dataclasses
    import subprocess
    git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True).stdout.strip()
    print(f"task {args.task_id}/{args.num_tasks}: {len(rows)} rows, config={mcfg.name} "
          f"(hash {mcfg.content_hash()}), memory={mem.name}, out_name={out_name}, "
          f"git={git_rev}, manifest={args.manifest}, "
          f"out_root={args.out_root}, seed_arg={args.seed}, "
          f"devices={jax.devices()}", flush=True)
    print("CONFIG " + json.dumps(dataclasses.asdict(mcfg), sort_keys=True), flush=True)
    print("MEMORY " + json.dumps(dataclasses.asdict(mem), sort_keys=True), flush=True)

    t_start = time.time()
    done = skipped = 0
    times = []
    seed = mcfg.seed if args.seed < 0 else args.seed

    def write_window(sid, variant, meta, results, out_arrays, traces, dt):
        nonlocal done
        results.update({"stim_id": sid, "variant": variant, "bundle_meta": meta,
                        "wall_s": dt})
        rdir = worklist.result_dir(args.out_root, out_name, variant, sid)
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

    def write_error(sid, variant, e):
        nonlocal done
        rdir = worklist.result_dir(args.out_root, out_name, variant, sid)
        atomic_write(rdir / "results.json", lambda tmp: Path(tmp).write_text(
            json.dumps({"stim_id": sid, "variant": variant, "error": str(e),
                        "config": mcfg.name, "memory": mem.name})))
        print(f"ERROR {sid}/{variant}: {e}", flush=True)
        done += 1

    if args.batch_windows > 1:
        from experiments.sfm.batched import infer_windows_batched
        from experiments.sfm import windows as W
        pending = []
        for sid, variant in rows:
            if worklist.is_done(args.out_root, out_name, variant, sid):
                skipped += 1
                continue
            bpath = bundles.bundle_path(cfg.BUNDLES_DIR, sid, variant)
            if not bpath.exists():
                print(f"MISSING bundle {bpath} — skipping", flush=True)
                continue
            pending.append((sid, variant, bpath))
        # windows of different lengths cannot stack: group chunks by T
        # (stable sort keeps the claim order within each T group)
        pending.sort(key=lambda r: len(W.window_frames(r[1])))
        B = args.batch_windows
        chunks = []
        i = 0
        while i < len(pending):
            T0 = len(W.window_frames(pending[i][1]))
            j = i
            while j < len(pending) and j - i < B and \
                    len(W.window_frames(pending[j][1])) == T0:
                j += 1
            chunks.append(pending[i:j])
            i = j
        for chunk in chunks:
            loaded = [bundles.load_bundle(p) for _, _, p in chunk]
            t0 = time.time()
            entries = infer_windows_batched([a for a, _ in loaded], mcfg,
                                            mem, seed=seed)
            dt = (time.time() - t0) / len(chunk)
            times.extend([dt] * len(chunk))
            for (sid, variant, _), (_, meta), entry in zip(chunk, loaded, entries):
                if entry[0] == "error":
                    write_error(sid, variant, entry[1])
                else:
                    _, results, out_arrays, traces = entry
                    write_window(sid, variant, meta, results, out_arrays, traces, dt)
        print(f"task done: {done} inferred, {skipped} skipped, "
              f"median {np.median(times) if times else float('nan'):.1f}s/window, "
              f"total {time.time()-t_start:.0f}s", flush=True)
        return

    for sid, variant in rows:
        # A manifest row (sid, "video"/"videoX") = that video's segments run IN
        # ORDER with the memory carry handed across segment boundaries
        # (mem.handoff decides whether/how the carried state is adopted). One row
        # per video keeps a video's segments in one worker by construction.
        # "video" = the blind 4-frame tiles seg0..seg5; "videoX" = the
        # metadata-aligned hold1/m0/m1/m2/hold2 (boundaries at motion onset and
        # offset, per the MWorks protocol + generation notebook).
        if variant in ("video", "videoX", "videoY"):
            from experiments.sfm import windows as W
            segs = (list(W.EXP2_SEGMENTS) if variant == "videoY"
                    else list(W.EXP_SEGMENTS) if variant == "videoX"
                    else [f"seg{i}" for i in range(6)])
            if all(worklist.is_done(args.out_root, out_name, sv, sid) for sv in segs):
                skipped += 1
                continue
            # static_bi: the static HEAD shows the same pose as the first moving
            # frames, so run the first moving-init segment FIRST and hand its
            # frame-0 grouping backward into the head statics (nearest first);
            # then continue forward as in 'static'.
            order = list(segs)
            head_carry = None
            if mem.handoff in ("static_bi", "static_pred"):
                inits = {}
                for sv in segs:
                    bp = bundles.bundle_path(cfg.BUNDLES_DIR, sid, sv)
                    if bp.exists():
                        a, _ = bundles.load_bundle(bp)
                        inits[sv] = float(np.asarray(a["motion_valid"])[0].mean())
                moving = [sv for sv in segs if inits.get(sv, 0.0) >= 0.08]
                if moving:
                    first = moving[0]
                    i0 = segs.index(first)
                    order = [first] + segs[:i0][::-1] + segs[i0 + 1:]
            carry = None
            src_trusted = False   # has the carried grouping ever seen real motion?
            for sv in order:
                bpath = bundles.bundle_path(cfg.BUNDLES_DIR, sid, sv)
                if not bpath.exists():
                    print(f"MISSING bundle {bpath} — skipping video {sid}", flush=True)
                    carry = None
                    continue
                arrays, meta = bundles.load_bundle(bpath)
                frame_motion = np.asarray(arrays["motion_valid"]).mean(axis=1)
                t0 = time.time()
                try:
                    if mem.is_off():   # degenerate use: independent segments
                        results, out_arrays, traces = infer_window(arrays, mcfg,
                                                                   seed=seed)
                    elif mem.handoff in ("static", "static_bi", "static_pred"):
                        # Motion-trust policy (static/static_bi): the assess-score
                        # guard cannot tell good grouping from bad when velocity
                        # evidence is weak (it accepted static-head garbage into
                        # moving segments, -0.46 on seg2-3), so adopt the carried
                        # state ONLY into static-init segments (whose fresh inits
                        # are near-chance) and only from motion-trusted sources.
                        # static_pred: offer wherever a carry exists (both
                        # directions, no trust filter) and let the marginal-data-
                        # likelihood gate inside infer_window_mem decide adoption
                        # — the direct test of grouping-aware acceptance.
                        tgt_static = bool(frame_motion[0] < 0.08)
                        backward = (head_carry is not None
                                    and segs.index(sv) < segs.index(order[0]))
                        cin = head_carry if backward else carry
                        if mem.handoff == "static_pred":
                            offer = cin is not None
                            mem_seg = dataclasses.replace(
                                mem,
                                handoff="static_pred" if offer else "none")
                        else:
                            trusted = True if backward else src_trusted
                            offer = bool(cin is not None and trusted
                                         and tgt_static)
                            mem_seg = dataclasses.replace(
                                mem, handoff="always" if offer else "none")
                        results, out_arrays, traces, cout = infer_window_mem(
                            arrays, mcfg, mem_seg, seed=seed,
                            carry_in=cin if offer else None)
                        adopted = (bool(results.get("handoff_used", False))
                                   if mem.handoff == "static_pred" else offer)
                        results["handoff_offered"] = bool(offer)
                        results["handoff_used"] = adopted
                        results["handoff_backward"] = bool(backward)
                        if backward:
                            # chain head rescues backward, nearest-first
                            head_carry = {"state": cout["state0"],
                                          "key": cout["key"],
                                          "roi_mask": cout["roi_mask_first"]}
                        else:
                            carry = cout
                            src_trusted = bool(frame_motion.max() > 0.08) or \
                                (adopted and src_trusted)
                            if (sv == order[0]
                                    and (mem.handoff == "static_pred"
                                         or (mem.handoff == "static_bi"
                                             and src_trusted))):
                                head_carry = {"state": cout["state0"],
                                              "key": cout["key"],
                                              "roi_mask": cout["roi_mask_first"]}
                    else:
                        results, out_arrays, traces, carry = infer_window_mem(
                            arrays, mcfg, mem, seed=seed, carry_in=carry)
                except Exception as e:
                    write_error(sid, sv, e)
                    carry = None
                    continue
                dt = time.time() - t0
                times.append(dt)
                write_window(sid, sv, meta, results, out_arrays, traces, dt)
            continue
        if worklist.is_done(args.out_root, out_name, variant, sid):
            skipped += 1
            continue
        bpath = bundles.bundle_path(cfg.BUNDLES_DIR, sid, variant)
        if not bpath.exists():
            print(f"MISSING bundle {bpath} — skipping", flush=True)
            continue
        arrays, meta = bundles.load_bundle(bpath)
        t0 = time.time()
        try:
            if mem.is_off():
                results, out_arrays, traces = infer_window(arrays, mcfg, seed=seed)
            else:
                results, out_arrays, traces, _ = infer_window_mem(
                    arrays, mcfg, mem, seed=seed)
        except Exception as e:  # degenerate windows etc.: record + move on
            write_error(sid, variant, e)
            continue
        dt = time.time() - t0
        times.append(dt)

        if args.expect_warm and done == 0 and dt > 600:
            print(f"ABORT: first window took {dt:.0f}s with --expect-warm "
                  f"(XLA cache miss?)", flush=True)
            sys.exit(3)

        write_window(sid, variant, meta, results, out_arrays, traces, dt)

    print(f"task done: {done} inferred, {skipped} skipped, "
          f"median {np.median(times) if times else float('nan'):.1f}s/window, "
          f"total {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
