"""
Deep check of real segment outputs (bundles + inference results + latent traces).
Run after preprocessing and run_sfm on the segcheck manifest:

  uv run python experiments/sfm/tests/check_seg_outputs.py \
      --manifest manifests/segcheck.tsv --config sfm_v2 --out-root results/windows
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.sfm import sfm_config as cfg
from experiments.sfm import windows as W
from experiments.sfm import bundles, worklist
from preprocessing.sfm.derive_masks import load_mask_grid

LATENTS = {"hyperblob_weights", "hyperblob_means", "hyperblob_covs",
           "hyperblob_trans_vels", "hyperblob_rot_vels",
           "blob_hyperblob_assignments", "blob_weights", "blob_means",
           "blob_covs", "blob_vel_means", "blob_vel_covs",
           "datapoint_assignments", "scores"}


def check_bundle(sid, variant):
    arrays, meta = bundles.load_bundle(bundles.bundle_path(cfg.BUNDLES_DIR, sid, variant))
    frames = W.window_frames(variant)
    T = len(frames) - 1
    assert arrays["points_3d"].shape[0] == T, f"{sid}/{variant}: T={arrays['points_3d'].shape[0]}"
    assert meta["evidence_gate"] is True, f"{sid}/{variant}: segments must be gated"
    assert tuple(meta["window_frames"]) == frames
    want = load_mask_grid(sid)[[W.frame_to_mask_index(f) for f in frames]]
    assert np.array_equal(arrays["gt_masks"], want), \
        f"{sid}/{variant}: gt_masks do not match clamped mask-stack rows"
    frame_motion = arrays["motion_valid"].mean(axis=1)   # [T] valid-motion fraction
    return T, frame_motion


def check_traces(sid, variant, mcfg, out_root, T, frame_motion):
    rdir = worklist.result_dir(out_root, mcfg.name, variant, sid)
    res = json.loads((rdir / "results.json").read_text())
    assert not res.get("error"), f"{sid}/{variant}: inference error: {res.get('error')}"
    assert len(res["frames"]) == T
    tpath = rdir / "traces.npz"
    d = np.load(tpath)
    phases = {"f0_init"} | {f"f{k}_{p}" for k in range(1, T) for p in ("vel", "track")}
    got_phases = {k.split(".")[0] for k in d.files}
    assert got_phases == phases, f"{sid}/{variant}: phases {got_phases} != {phases}"
    for ph in phases:
        keys = {k.split(".", 1)[1] for k in d.files if k.startswith(ph + ".")}
        assert keys == LATENTS, f"{sid}/{variant}/{ph}: latents {keys ^ LATENTS}"
    sweeps = {"f0_init": mcfg.init_sweeps}
    sweeps.update({f"f{k}_vel": mcfg.vel_sweeps for k in range(1, T)})
    sweeps.update({f"f{k}_track": mcfg.track_sweeps for k in range(1, T)})
    L, H, N = mcfg.n_blobs, mcfg.n_hyperblobs, cfg.N_DATAPOINTS
    for ph, n_sw in sweeps.items():
        kept = int(np.ceil(n_sw / mcfg.trace_thin))
        da = d[f"{ph}.datapoint_assignments"]
        assert da.shape == (kept, N) and da.dtype == np.int16, f"{ph}: {da.shape} {da.dtype}"
        assert d[f"{ph}.blob_means"].shape == (kept, L, 3)
        assert d[f"{ph}.blob_means"].dtype == np.float16
        assert d[f"{ph}.hyperblob_covs"].shape == (kept, H, 3, 3)
        assert d[f"{ph}.blob_hyperblob_assignments"].dtype == np.int8
        s = d[f"{ph}.scores"]
        assert s.shape == (n_sw,)
        assert not np.isnan(s).any(), f"{ph}: NaN scores"
        # a timestep with zero motion evidence (static frame pair) legitimately
        # assesses to -inf; a timestep with real motion must stay finite
        frame = 0 if ph == "f0_init" else int(ph[1:].split("_")[0])
        if frame_motion[frame] > 0.01:
            assert np.isfinite(s).all(), \
                f"{ph}: non-finite scores at a moving timestep ({frame_motion[frame]:.3f})"
        amax = int(da.max())
        assert amax <= L, f"{ph}: datapoint assignment {amax} > L={L}"
    return tpath.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="sfm_v2")
    ap.add_argument("--out-root", default=str(cfg.RESULTS_DIR / "windows"))
    args = ap.parse_args()
    all_cfgs = dict(cfg.CONFIGS)
    all_cfgs.update(cfg.pilot_config_grid())
    mcfg = all_cfgs[args.config]

    rows = worklist.read_manifest(args.manifest)
    sizes = []
    for sid, variant in rows:
        T, frame_motion = check_bundle(sid, variant)
        sizes.append(check_traces(sid, variant, mcfg, args.out_root, T, frame_motion))
    mb = np.array(sizes) / 1e6
    print(f"checked {len(rows)} windows: traces {mb.mean():.2f} MB/window "
          f"(max {mb.max():.2f}); projected 5040-window total {mb.mean()*5040/1e3:.1f} GB")
    print("check_seg_outputs: PASS")


if __name__ == "__main__":
    main()
