"""
Memory-mechanism configuration, kept OFF SfmModelConfig so the recorded content
hash of sfm_v2 (and every phase-1 result) is untouched. Results for a memory run
are routed under "<model_config>+<memory_name>" (e.g. results/windows/sfm_v2+sticky2/).
"""
import hashlib
import json
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class MemoryConfig:
    name: str
    # sticky assignments: bonus kappa*one_hot(prev_assign) in the move logits;
    # kappa_sel augments the in-scan selection score by kappa_sel*sum(z == prev)
    # (-1.0 = follow kappa). Init frame always runs kappa = 0.
    kappa: float = 0.0
    kappa_sel: float = -1.0
    # sufficient-stat filtering: 0 = off; else the between-frame blend weight
    filter_lambda: float = 0.0
    process_noise_q: float = 1e-4       # Sigma0 floor for blob-mean priors
    vel_process_noise_q: float = 1e-5   # Sigma0 floor for velocity-mean priors
    # False = filter positions/shapes only; velocity moves stay stock. Velocity
    # memory entrenches stale motion at cessation (frames where motion stops):
    # the filtered prior says "still moving" while the data says zero.
    filter_velocity: bool = True
    # True = scale the VELOCITY filter strength by the current frame's motion
    # evidence (valid-motion fraction / vel_evidence_floor, capped at 1): full
    # velocity memory while motion persists, stock velocities at cessation.
    adaptive_vel: bool = False
    vel_evidence_floor: float = 0.08
    # with adaptive_vel: True = binary gate (full velocity memory iff the frame's
    # motion fraction clears the floor, zero below — a linear scale still leaves
    # enough stale-velocity prior at cessation to cost ~-0.1 there)
    vel_gate_binary: bool = False
    # evidence-conditioned assignment freezing: extra sticky strength
    # freeze_kappa*(1 - evidence) added to kappa (and to the selection bonus).
    # At zero motion evidence the anchor grouping is strongly held — in SFM
    # textured statics the object is invisible (uniform texture), so ONLY a
    # carried grouping can bridge the holds; without this it decays to prior.
    # At full evidence the term vanishes and the chain is the plain sticky one.
    freeze_kappa: float = 0.0
    # forward-backward: 2 = rerun forward with next-frame-informed priors
    smooth_passes: int = 1
    # multi-particle SMC (1 = single chain)
    n_particles: int = 1
    # cross-segment state handoff: none | always | guarded
    handoff: str = "none"

    def kappa_sel_resolved(self) -> float:
        return self.kappa if self.kappa_sel < 0 else self.kappa_sel

    def is_off(self) -> bool:
        return (self.kappa == 0.0 and self.filter_lambda == 0.0
                and self.freeze_kappa == 0.0
                and self.smooth_passes == 1 and self.n_particles == 1
                and self.handoff == "none")

    def content_hash(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True)
                            .encode()).hexdigest()[:10]

    def result_name(self, model_config_name: str) -> str:
        return model_config_name if self.is_off() else f"{model_config_name}+{self.name}"


MEM_OFF = MemoryConfig(name="off")


def memory_config_grid():
    """Bake-off grid, cheapest-first (plan Workstream 4). Sticky sweep first; the
    filtering sweep composes with the sticky winner (same compiled program)."""
    grid = [MEM_OFF]
    # matched-baseline control: takes the memory code path (and current hypers
    # guards) with a negligible kappa, so paired bake-off deltas share everything
    # but the mechanism under test
    grid.append(MemoryConfig(name="null", kappa=1e-6, kappa_sel=0.0))
    for k in (0.5, 1.0, 2.0, 4.0):
        grid.append(MemoryConfig(name=f"sticky{k:g}", kappa=k))
    for k in (1.0, 2.0):
        for lam in (0.3, 0.5, 0.7):
            grid.append(MemoryConfig(name=f"sticky{k:g}_filt{lam:g}",
                                     kappa=k, filter_lambda=lam))
    for lam in (0.3, 0.5, 0.7, 0.85):
        grid.append(MemoryConfig(name=f"filt{lam:g}", filter_lambda=lam))
        grid.append(MemoryConfig(name=f"filt{lam:g}p", filter_lambda=lam,
                                 filter_velocity=False))
        grid.append(MemoryConfig(name=f"filt{lam:g}a", filter_lambda=lam,
                                 adaptive_vel=True))
        grid.append(MemoryConfig(name=f"filt{lam:g}b", filter_lambda=lam,
                                 adaptive_vel=True, vel_gate_binary=True))
    # cross-segment handoff (whole-video memory), composed with the in-window
    # mechanisms; "_hoA" = always adopt the carried state, "_hoG" = guarded
    for base_name, k, lam in (("ho", 0.0, 0.0), ("sticky1_ho", 1.0, 0.0),
                              ("sticky1_filt0.5_ho", 1.0, 0.5)):
        for mode, suf in (("guarded", "G"), ("always", "A"), ("static", "S")):
            grid.append(MemoryConfig(name=f"{base_name}{suf}", kappa=k,
                                     filter_lambda=lam, handoff=mode))
    # bidirectional static rescue on the adaptive-velocity filter winner
    for lam in (0.5, 0.7):
        grid.append(MemoryConfig(name=f"filt{lam:g}b_hoB", filter_lambda=lam,
                                 adaptive_vel=True, vel_gate_binary=True,
                                 handoff="static_bi"))
    # evidence-conditioned assignment freezing on the winner (kappa_frz sweep),
    # and the predictive-acceptance handoff ("static_pred": motion-trust offer
    # set, accept/reject by held-out-free marginal data likelihood)
    for kf in (20.0, 100.0):
        grid.append(MemoryConfig(name=f"filt0.7b_hoB_frz{kf:g}",
                                 filter_lambda=0.7, adaptive_vel=True,
                                 vel_gate_binary=True, handoff="static_bi",
                                 freeze_kappa=kf))
        grid.append(MemoryConfig(name=f"filt0.7b_hoP_frz{kf:g}",
                                 filter_lambda=0.7, adaptive_vel=True,
                                 vel_gate_binary=True, handoff="static_pred",
                                 freeze_kappa=kf))
    # long-horizon (mfull, 12 transitions) escalation rungs: forward-backward
    # smoother (needs filter_lambda > 0 — the smoothing priors ride the
    # filtering path) and SMC (n_particles > 1, vmapped chains, systematic
    # resampling on assignment-marginalized predictive weights)
    for lam in (0.5, 0.7):
        grid.append(MemoryConfig(name=f"filt{lam:g}b_sm2", filter_lambda=lam,
                                 adaptive_vel=True, vel_gate_binary=True,
                                 smooth_passes=2))
    for P in (4, 8, 16):
        grid.append(MemoryConfig(name=f"filt0.7b_smc{P}", filter_lambda=0.7,
                                 adaptive_vel=True, vel_gate_binary=True,
                                 n_particles=P))
    # pure mechanisms for the sv0.01-coupled chain (velocity filtering is
    # superseded there — the transform prior regularizes velocities): freeze
    # targets the zero-evidence cessation frame; SMC adds hypothesis diversity
    for kf in (20.0, 100.0):
        grid.append(MemoryConfig(name=f"frz{kf:g}", freeze_kappa=kf))
    for P in (8, 16):
        grid.append(MemoryConfig(name=f"smc{P}", n_particles=P))
    return {m.name: m for m in grid}
