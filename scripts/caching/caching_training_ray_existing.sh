#!/bin/bash

# StyleDrive Dataset Caching Script (Ray Existing Cluster)
# This script connects to an already running Ray cluster

# Ensure environment variables are set
if [ -z "$NAVSIM_DEVKIT_ROOT" ]; then
    echo "Error: NAVSIM_DEVKIT_ROOT is not set. Please run: source ./setup_env.sh"
    exit 1
fi

if [ -z "$NAVSIM_EXP_ROOT" ]; then
    echo "Error: NAVSIM_EXP_ROOT is not set. Please run: source ./setup_env.sh"
    exit 1
fi

echo "Using NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT"
echo "Using NAVSIM_EXP_ROOT: $NAVSIM_EXP_ROOT"
echo "Using existing Ray cluster"

# Check if Ray is running
if ! ray status >/dev/null 2>&1; then
    echo "Error: Ray cluster is not running. Please start Ray first:"
    echo "  ray start --head --disable-usage-stats"
    exit 1
fi

echo "Ray cluster is running:"
ray status

# Create cache directory
mkdir -p "$NAVSIM_EXP_ROOT/training_cache"

# Run caching command with existing Ray cluster
python $NAVSIM_DEVKIT_ROOT/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_style_agent \
    experiment_name=training_diffusiondrive_style_agent \
    train_test_split=styletrain \
    cache_path=$NAVSIM_EXP_ROOT/training_cache \
    worker=ray_existing_cluster