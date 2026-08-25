"""Manifest construction and work claiming for SLURM array tasks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.sfm import sfm_config as cfg
from experiments.sfm import windows as W


def pilot_stim_ids():
    """Stratified 18-video pilot: 3 objects x 6 textures, spread over sizes/viewpoints."""
    ids = []
    for oi, obj in enumerate((0, 7, 14)):
        for ti in range(len(cfg.TEXTURES)):
            combo = oi * len(cfg.TEXTURES) + ti
            size_idx = combo % 3
            viewpoint = combo % cfg.N_VIEWPOINTS
            ids.append(cfg.stim_id_of(size_idx, obj, ti, viewpoint))
    return sorted(ids)


def build_manifest(path, stim_ids, variants):
    rows = [(sid, v) for sid in sorted(stim_ids) for v in variants]
    path = cfg.assert_writable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{sid}\t{v}\n" for sid, v in rows))
    return rows


def read_manifest(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            sid, v = line.split("\t")
            rows.append((int(sid), v))
    return rows


def claim(rows, task_id, num_tasks):
    return rows[task_id::num_tasks]


def result_dir(out_root, config_name, variant, stim_id) -> Path:
    return Path(out_root) / config_name / variant / f"{stim_id:04d}"


def is_done(out_root, config_name, variant, stim_id) -> bool:
    return (result_dir(out_root, config_name, variant, stim_id) / "results.json").exists()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", choices=["pilot", "full", "warmup", "large27"],
                    required=True)
    ap.add_argument("--variants", default="tiledA,tiledB,centered,stride2")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.variants == "all":
        variants = list(W.PILOT_VARIANTS)
    elif args.variants == "exp2":         # metadata-aligned + 1-frame overlap (preferred)
        variants = list(W.EXP2_SEGMENTS)
    elif args.variants == "exp":          # metadata-aligned, 19/24 frames evaluated
        variants = list(W.EXP_SEGMENTS)
    elif args.variants == "segments":     # legacy blind tiles
        variants = sorted(W.SEGMENT_VARIANTS)
    else:
        variants = args.variants.split(",")
    # sanity gate: the stim_id arithmetic must match the CSV before any manifest exists
    bad = cfg.verify_metadata_formula(cfg.load_meta_rows())
    if bad:
        raise SystemExit(f"metadata formula mismatches for stim_ids {bad[:10]}...")

    if args.make == "pilot":
        ids = pilot_stim_ids()
    elif args.make == "warmup":
        ids = pilot_stim_ids()[:1]
    elif args.make == "large27":
        # all 840 large-FOV (27 deg = size_idx 2) videos
        ids = [i for i in range(2 * 840, 3 * 840)]
    else:
        ids = [i for i in range(cfg.N_STIMULI)]

    out = Path(args.out) if args.out else cfg.WORK_ROOT / "manifests" / f"{args.make}.tsv"
    rows = build_manifest(out, ids, variants)
    # also drop the plain stim-id list for the preprocessing stage
    ids_file = out.with_suffix(".stims.txt")
    cfg.assert_writable_path(ids_file).write_text("\n".join(str(i) for i in ids) + "\n")
    print(f"{out}: {len(rows)} rows ({len(ids)} videos x {variants})")


if __name__ == "__main__":
    main()
