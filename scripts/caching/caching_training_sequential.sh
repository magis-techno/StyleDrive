#!/bin/bash

# StyleDrive Dataset Caching Script (Sequential Worker)
# This script caches the training dataset using a single-threaded sequential worker
# Most stable option, suitable for systems with Ray compatibility issues

# Ensure environment variables are set
if [ -z "$NAVSIM_DEVKIT_ROOT" ]; then
    echo "Error: NAVSIM_DEVKIT_ROOT is not set. Please run: source ./env_vars.sh"
    exit 1
fi

if [ -z "$NAVSIM_EXP_ROOT" ]; then
    echo "Error: NAVSIM_EXP_ROOT is not set. Please run: source ./env_vars.sh"
    exit 1
fi

echo "Using NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT"
echo "Using NAVSIM_EXP_ROOT: $NAVSIM_EXP_ROOT"
echo "Using sequential worker (single-threaded, most stable)"

# Create cache directory
mkdir -p "$NAVSIM_EXP_ROOT/training_cache"

# Run caching command with sequential worker
python $NAVSIM_DEVKIT_ROOT/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_style_agent \
    experiment_name=training_diffusiondrive_style_agent \
    train_test_split=styletrain \
    cache_path=$NAVSIM_EXP_ROOT/training_cache \
    worker=sequential 