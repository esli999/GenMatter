import os
import json
import time
import jax.numpy as jnp
from jax.random import key as jkey

import numpy as np
import pickle
from tqdm import tqdm

from tensorflow_probability.substrates.jax import distributions as tfd

from genparticles.datatypes import *
from genparticles.model_3d import *
from genparticles.inference import *
from genparticles.dataloader import *
from genparticles.utils import *
from genparticles.evaluation import *

model_jsimulate = jax.jit(HDGMM_model_3d.simulate)
model_jimportance = jax.jit(HDGMM_model_3d.importance)

######################################################
## CONFIGURATION START
######################################################

EXPERIMENT_NAME = "Davis_Benchmark_Results"
# Define the experiment save directory
EXPERIMENT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"Experiment save directory: {EXPERIMENT_SAVE_DIR}")

HYPERPARAMS = {
    "number_of_blobs": 500,
    "number_of_hyperblobs": 9,
    "num_seeded_runs": 5,
}

SAVE_DATA = True

# List of 33 available stimuli for benchmarking
DAVIS_STIMULI = [
    "dog-agility", "mallard-fly", "rallye", "soccerball", "varanus-cage",
    "breakdance", "bear", "koala", "dance-twirl", "drift-straight",
    "parkour", "breakdance-flare", "lucia", "drift-chicane", "car-roundabout",
    "blackswan", "boat", "dog", "elephant", "goat",
    "cows", "libby", "bus", "car-shadow", "flamingo",
    "camel", "hike", "car-turn", "mallard-water", "dance-jump",
    "rhino", "rollerblade", "drift-turn"
]


#SET PATH TO STIMULI DATA
DAVIS_STIMULI_PATH = ""
DAVIS_SEGMASKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/davis_segmasks")

######################################################
## CONFIGURATION END
######################################################


def run_hdgmm_benchmark(stimulus, hyperparams, experiment_name, experiment_dir):
    """
    Run HDGMM benchmark for a single stimulus with multiple random seeds.
    
    Args:
        stimulus: Name of the stimulus to process
        hyperparams: Dictionary containing hyperparameters for the experiment
        experiment_name: Name of the experiment
        experiment_dir: Directory where experiment data should be saved
        
    Returns:
        tuple: (tracking_data, random_seeds, img_dims) for the stimulus
    """
    # Expand hyperparameters from the dictionary
    number_of_blobs = hyperparams["number_of_blobs"]
    number_of_hyperblobs = hyperparams["number_of_hyperblobs"]
    num_seeded_runs = hyperparams["num_seeded_runs"]
    
    print(f"\nStarting benchmark for stimulus: {stimulus}")

    # Extract feature point data
    tracked_points, tracked_motion_vectors, num_data_tsteps, img_dims = extract_3d_points_and_motion_vectors_data(DAVIS_STIMULI_PATH, stimulus)
    first_frame_seg = get_segmentation_mask(stimulus, 0, DAVIS_SEGMASKS_PATH, img_dims=img_dims, flatten=True)

    kmeans_chm, roi_blob_indices, roi_hyperblob_indices = make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob(tracked_points, number_of_blobs, number_of_hyperblobs, segmentation_mask = first_frame_seg, motion_vectors = tracked_motion_vectors)

    # Set up Gibbs sampling dials
    GIBBS_DIALS = {
        "blob_weights": True,
        "hyperblob_weights": False,
        "blob_assignments": True,
        "hyperblob_assignments": False,
        "hyperblob_covs": False,
        "blob_covs": True,
        "blob_vel_covs": True,
        "blob_vel_means": True,
        "hyperblob_means": False,
        "blob_means": True,
        'hyperblob_rot_vels': False,
        'hyperblob_trans_vels': False
    }

    # hyperparams are scaled to the length scale of the data
    num_datapoints = kmeans_chm['datapoints', 'datapoint_positions'].shape[0]
    num_blobs = kmeans_chm['blobs', 'hyperblob_assignments'].shape[0]
    num_hyperblobs = kmeans_chm['hyperblobs', 'hyperblob_means'].shape[0]
    empirical_mu_H = jnp.median(kmeans_chm['datapoints', 'datapoint_positions'], axis = 0)
    empirical_sigma_H = (10*0.5)**2
    empirical_Psi_B = jnp.median(kmeans_chm['blobs', 'blob_covs'][roi_blob_indices], axis = 0)
    empirical_Psi_H = jnp.median(kmeans_chm['hyperblobs', 'hyperblob_covs'][roi_hyperblob_indices], axis = 0)
    empirical_Psi_V = jnp.median(kmeans_chm['blobs', 'blob_vel_covs'][roi_blob_indices], axis = 0)
    mean_blobs_per_roi_hyperblob = jnp.sum(jnp.isin(kmeans_chm['blobs', 'hyperblob_assignments'], roi_hyperblob_indices)) / len(roi_hyperblob_indices)
    empirical_nu_H = f_(int(mean_blobs_per_roi_hyperblob))
    mean_points_per_roi_blob = jnp.sum(jnp.isin(kmeans_chm['datapoints', 'blob_assignments'], roi_blob_indices)) / len(roi_blob_indices)
    empirical_nu_B = empirical_nu_V = f_(int(mean_points_per_roi_blob))

    # Create hyperparameters
    hypers = HDGMM_Hyperparams.create(
        outlier_prob=f_(5e0),
        outlier_velocity_gamma_shape=f_(5.0),
        outlier_velocity_gamma_rate=f_(1.0),
        alpha=f_(1.0),
        beta=f_(1.0),
        mu_H=empirical_mu_H,
        sigma_H=empirical_sigma_H,
        nu_H=empirical_nu_H,
        Psi_H=empirical_Psi_H,
        nu_B=empirical_nu_B,
        Psi_B=empirical_Psi_B,
        sigma_V=f_(10e14),
        nu_V=empirical_nu_V,
        Psi_V=empirical_Psi_V,
        translation_gaussian_scale=snp(f_(0.2)),
        translation_max_radius=snp(0.35),
        translation_num_radii_cells=snp(15),
        translation_theta_step_deg=snp(15),
        rotation_vmf_kappa=snp(f_(100)),
        rotation_angle_max_deg=snp(25),
        rotation_angle_step_deg=snp(0.375),
        n_hyperblobs=num_hyperblobs,
        n_blobs=num_blobs,
        n_datapoints=num_datapoints,
    )

    # Initialize lists to store results
    multiple_tracking_data = []

    # Run multiple seeds
    for j in range(num_seeded_runs):
        print(f"Running seed {j+1} of {num_seeded_runs} for {stimulus}")

        key = jkey(np.random.randint(1e9))

        # Initialize model
        key, key_importance = jax.random.split(key)
        init_tr, _ = model_jimportance(key_importance, kmeans_chm, (hypers,))
        init_hdgmm_state = init_tr.get_retval()

        # Run initial Gibbs sweeps
        key, init_gibbs_key = jax.random.split(key)
        gibbs_wtrs = hdgmm_full_gibbs(init_gibbs_key, init_hdgmm_state, 15, GIBBS_DIALS, use_weighted_blobs=True, num_gibbs_inner_loops=1)

        # Get final state and clear memory
        init_hdgmm_state = gibbs_wtrs[-1].retval
        
        # Run tracking
        key, tracking_key = jax.random.split(key)
        tracking_wtrs = hdgmm_tracking_gibbs(tracking_key, init_hdgmm_state, tracked_points, tracked_motion_vectors)
        # Clear memory
        
        # Extract only the necessary data for evaluation
        tracking_data = []
        for frame_idx in range(len(tracking_wtrs)):
            # Extract n_blobs and blob_assignments as a dictionary
            frame = tracking_wtrs[frame_idx]
            frame_data = {
                'n_blobs': frame.retval.hypers.n_blobs,
                'blob_assignments': frame.retval.datapoints_state.blob_assignments.copy()
            }
            tracking_data.append(frame_data)
        
        # Store tracking results
        multiple_tracking_data.append(tracking_data)
        

    return multiple_tracking_data, img_dims

# Process each stimulus and collect results
all_results = {}
experiment_start_time = time.time()
for i, stimulus in enumerate(DAVIS_STIMULI):
    print(f"\nProcessing stimulus {i+1}/{len(DAVIS_STIMULI)}: {stimulus}")
    try:
        multiple_tracking_data, img_dims = run_hdgmm_benchmark(stimulus, HYPERPARAMS, EXPERIMENT_NAME, EXPERIMENT_SAVE_DIR)
        all_results[stimulus] = multiple_tracking_data
    except Exception as e:
        print(f"Error processing stimulus {stimulus}: {e}")
        continue

experiment_end_time = time.time()
total_minutes = (experiment_end_time - experiment_start_time) / 60
print(f"Experiment completed in {total_minutes:.2f} minutes for {len(DAVIS_STIMULI)} stimuli")

# Run evaluation and visualization after all stimuli are processed
evaluate_tracking_results(
    all_results, 
    annotations_path=DAVIS_SEGMASKS_PATH,
    save_data=SAVE_DATA,
    output_dir=os.path.join(EXPERIMENT_SAVE_DIR, EXPERIMENT_NAME),
    experiment_name=EXPERIMENT_NAME
)
