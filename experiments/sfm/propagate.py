"""Jitted per-frame state propagation.

Replaces run_gestalt.py's Python loops + full choicemap rebuild + importance call.
Because the choicemap there is fully constrained, importance is deterministic and a
direct `state.replace` is exactly equivalent for every consumed quantity. One benign
deviation: hyperblobs with zero assigned blobs get mean+translation here, where the
original zeroed them — no Gibbs move ever consumes an empty hyperblob's mean before
it is resampled from a posterior that ignores the current value.
"""
import jax
import jax.numpy as jnp


@jax.jit
def propagate_state(state, points_t, motion_t):
    a = state.blobs_state.hyperblob_assignments            # (L,)
    R = state.hyperblobs_state.hyperblob_rot_vels          # (K, 3, 3)
    t = state.hyperblobs_state.hyperblob_trans_vels        # (K, 3)
    muH = state.hyperblobs_state.hyperblob_means           # (K, 3)

    Rb, tb, muHb = R[a], t[a], muH[a]
    rel = state.blobs_state.blob_means - muHb
    new_blob_means = muHb + tb + jnp.einsum("lij,lj->li", Rb, rel)

    return state.replace({
        "hyperblobs_state": {"hyperblob_means": muH + t},
        "blobs_state": {"blob_means": new_blob_means},
        "datapoints_state": {
            "datapoint_positions": points_t,
            "datapoint_vels": motion_t,
        },
    })
