# GenParticles_NeurIPS
Anonymized Supplementary Materials Submission for GenParticles (NeurIPS 2025)

## Setup and Benchmark Instructions

**Requirements:**
- CUDA 12.4 must be installed on your system (with a compatible driver).
- You need a single NVIDIA GPU with at least 24GB of Memory.
- If a different CUDA version is installed, it must be compatible with JAX and GenJAX versions from `requirements.txt`

### 1. Create and Activate Conda Environment

```
conda create -n genparticles python=3.11
conda activate genparticles
pip install -r requirements.txt
```

### 2. Extract Depth and Motion Estimates

Install VideoDepthAnything (https://github.com/DepthAnything/Video-Depth-Anything), and use davis_script.sh to preprocess all the depth files. run.py from VideoDepthAnything can be left unmodified. 

Run preprocessing/motion_extraction_3d/davis_motion_extraction.py to get npz files containing 3D motion information for each of the davis videos. This script will use an implementation of RAFT on PyTorch Hub. 

### 3. Run DAVIS Benchmark
```
python davis_benchmark.py 
```

### 4. Run Psychophysics Benchmark
```
python preprocessing/random_dot_kinematograms/RDK_extraction.py
python psychophysics_benchmark.py
```

### 5. Run Baseline Comparisons 

SpaTracker: After installing SpaTracker (https://github.com/henry123-boy/SpaTracker), update baseline_experiments/run_spatracker_baseline.sh script and baseline_experiments/run_spatracker.py to point to the downloaded DAVIS assets folders. To run SpaTracker with our provided script, the individual frames of the video will have to be encoded into an MP4 file. baseline_experiments/calculate_spatracker_accuracy.py will calculate statistics for SpaTracker's accuracy performance. 

CoTracker3: baseline_experiments/cotracker_accuracy.py is configured to download the offline model from PyTorch Hub, run it on the DAVIS videos, and calculate statistics. Update the assets folders and results folders to point to the folders on your machine. 


