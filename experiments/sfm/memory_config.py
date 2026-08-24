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
    for k in (0.5, 1.0, 2.0, 4.0):
        grid.append(MemoryConfig(name=f"sticky{k:g}", kappa=k))
    for k in (1.0, 2.0):
        for lam in (0.3, 0.5, 0.7):
            grid.append(MemoryConfig(name=f"sticky{k:g}_filt{lam:g}",
                                     kappa=k, filter_lambda=lam))
    for lam in (0.3, 0.5, 0.7):
        grid.append(MemoryConfig(name=f"filt{lam:g}", filter_lambda=lam))
    return {m.name: m for m in grid}
