from genparticles.psychophysics_utils import *
from jax.random import key as jkey
import numpy as np
from tqdm import tqdm
import jax
import os
import random
import json
from scipy import stats

# Load the stimulus configurations from JSON file
with open(os.path.join(os.path.dirname(__file__), "assets/RDK/RDK_configs.json"), "r") as f:
    STIMULI_CONFIGS = json.load(f)


genpm_psychophysics_results = {}
count = 0
for stim_id, stimulus in STIMULI_CONFIGS.items():
    # Ignore familiarization stimuli
    if 'fam' in stim_id:
        continue
    count += 1
    config_num = stimulus['config_num']
    start_frame = stimulus['start_frame']
    end_frame = stimulus['end_frame']
    probe_timestep = stimulus['probe_timestep']
    # Randomly assign red and green points with 50% probability
    if random.random() < 0.5:
        # 50% chance: red = point_1, green = point_2
        red_point = stimulus['point_1']
        green_point = stimulus['point_2']
    else:
        # 50% chance: red = point_2, green = point_1
        red_point = stimulus['point_2']
        green_point = stimulus['point_1']

    ground_truth = stimulus['ground_truth']

    npz_path = os.path.join(os.path.dirname(__file__), f"assets/RDK/config_{config_num}/data.npz")
    data = np.load(npz_path)

    init_key_int = np.random.randint(1e9)
    key = jkey(init_key_int)
    genpm_model_results = []
    for i in tqdm(range(10)):
        key, exp_key = jax.random.split(key)
        result = model_prediction_on_stimulus(exp_key, config_num, start_frame, end_frame, probe_timestep, red_point, green_point, num_runs=5)
        genpm_model_results.append(result)
        
    genpm_psychophysics_result = np.mean(np.concatenate(genpm_model_results)) * 100
    genpm_psychophysics_results[stim_id] = genpm_psychophysics_result

    print(f"stim_id: {stim_id}, Model Prediction: {genpm_psychophysics_result:.2f}%, Ground Truth: {ground_truth}")

human_results = { # These are the raw human results from the psychophysics experiment
    # Set 1
    's1_c1': 86.0,
    's1_c2': 92.0,
    's1_c4': 64.0,
    's1_c5': 82.0,
    's1_c6': 84.0,
    's1_c7': 12.0,
    's1_c8': 90.0,
    's1_c9': 14.0,
    's1_c10': 54.0,

    # Set 2
    's2_c1': 42.0,
    's2_c2': 86.0,
    's2_c4': 80.0,
    's2_c5': 78.0,
    's2_c6': 18.0,
    's2_c7': 38.0,
    's2_c8': 66.0,
    's2_c9': 30.0,
    's2_c10': 88.0,

    # Set 3
    's3_c1': 84.0,
    's3_c2': 18.0,
    's3_c4': 94.0,
    's3_c5': 90.0,
    's3_c6': 26.0,
    's3_c7': 44.0,
    's3_c8': 90.0,
    's3_c9': 36.0,
    's3_c10': 84.0,
}

# Print summary table
print("\nResults Summary:")
print("=" * 45)
print(f"{'Stimulus ID':12} {'Model Prediction':>15} {'Human Results':>15}")
print("-" * 45)

differences = []
model_vals = []
human_vals = []

for stim_id in sorted(genpm_psychophysics_results.keys()):
    if stim_id in human_results:
        model_pred = genpm_psychophysics_results[stim_id]
        human_res = human_results[stim_id]
        diff = model_pred - human_res
        differences.append(abs(diff))
        model_vals.append(model_pred)
        human_vals.append(human_res)
        print(f"{stim_id:12} {model_pred:15.2f} {human_res:15.2f}")

print("-" * 45)

# Calculate R-squared
correlation_matrix = np.corrcoef(model_vals, human_vals)
r_squared = correlation_matrix[0,1]**2

print(f"\nR-squared correlation: {r_squared:.3f}")