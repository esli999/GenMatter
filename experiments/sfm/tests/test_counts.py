"""Vectorized sample counts == the original pure-Python formulation. Run directly."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.sfm.counts import (hyperblob_counts, hyperblob_counts_reference,
                                    pixel_hyperblobs)


def main():
    rng = np.random.default_rng(7)
    for trial in range(5):
        S, G, L, K = 13, 16, 21, 4
        N = G * G
        ba = rng.integers(0, L + 1, size=(S, N))           # L = outlier index
        ha = rng.integers(0, K, size=(S, L))
        fast = hyperblob_counts(ba, ha, L, K, G)
        ref = hyperblob_counts_reference(ba, ha, L, K, G)
        assert np.array_equal(fast, ref), f"trial {trial}: counts mismatch"
        ph = pixel_hyperblobs(ba[0], ha[0], L)
        ph_ref = np.array([ha[0][b] if b < L else -1 for b in ba[0]])
        assert np.array_equal(ph, ph_ref), f"trial {trial}: pixel_hyperblobs mismatch"
    print("test_counts: PASS")


if __name__ == "__main__":
    main()
