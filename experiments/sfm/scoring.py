"""Jitted joint log-probability of a GenMatter_State (assess-based).

This is the model's own scorer (GenMatter_model_3d.assess) applied to a choicemap
rebuilt from state arrays — identical to run_gestalt.py's rescoring, but traceable
and O(N) per call, so it can run inside the Gibbs scan every sweep.
"""
import jax
from genjax import ChoiceMapBuilder as C

from genmatter.model_3d import GenMatter_model_3d


def state_to_choicemap(state):
    return (
        C["hyperblobs", "hyperblob_weights"].set(state.hyperblobs_state.hyperblob_weights)
        | C["hyperblobs", "hyperblob_means"].set(state.hyperblobs_state.hyperblob_means)
        | C["hyperblobs", "hyperblob_covs"].set(state.hyperblobs_state.hyperblob_covs)
        | C["hyperblobs", "hyperblob_trans_vels"].set(state.hyperblobs_state.hyperblob_trans_vels)
        | C["hyperblobs", "hyperblob_rot_vels"].set(state.hyperblobs_state.hyperblob_rot_vels)
        | C["blobs", "hyperblob_assignments"].set(state.blobs_state.hyperblob_assignments)
        | C["blobs", "blob_weights"].set(state.blobs_state.blob_weights)
        | C["blobs", "blob_means"].set(state.blobs_state.blob_means)
        | C["blobs", "blob_covs"].set(state.blobs_state.blob_covs)
        | C["blobs", "blob_vel_means"].set(state.blobs_state.blob_vel_means)
        | C["blobs", "blob_vel_covs"].set(state.blobs_state.blob_vel_covs)
        | C["datapoints", "blob_assignments"].set(state.datapoints_state.blob_assignments)
        | C["datapoints", "datapoint_positions"].set(state.datapoints_state.datapoint_positions)
        | C["datapoints", "datapoint_vels"].set(state.datapoints_state.datapoint_vels)
    )


def joint_logprob(state):
    """Traced scorer (used inside the Gibbs scan)."""
    weight, _ = GenMatter_model_3d.assess(state_to_choicemap(state), (state.hypers,))
    return weight


joint_logprob_jit = jax.jit(joint_logprob)
