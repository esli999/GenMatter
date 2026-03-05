#!/bin/bash

# Define the list of stimuli
DAVIS_STIMULI=(
    "varanus-cage"
    "breakdance" "bear" "dog-agility" "dance-twirl" "drift-straight"
    "parkour" "breakdance-flare" "lucia" "drift-chicane" "car-roundabout"
    "blackswan" "boat" "dog" "elephant" "goat"
    "cows" "libby" "bus" "car-shadow" "flamingo"
    "camel" "hike" "car-turn" "mallard-water" "dance-jump"
    "rhino" "rollerblade" "drift-turn"
    "koala" "mallard-fly" "rallye" "soccerball"
)

# Run the python script for each stimulus
for stimulus in "${DAVIS_STIMULI[@]}"; do
    echo "Processing $stimulus..."
    python demo_baseline.py \
        --downsample 0.25 \
        --vid_name REPLACE_WITH_PATH_TO_VIDEOS/$stimulus \
        --fps 1 \
        --grid_size 25 \
        --gpu 0 \
        --outdir REPLACE_WITH_PATH_TO_SAVE_RESULTS/
done