#!/bin/bash

# StyleDrive Dataset Caching Script (Thread Pool Worker)
# This script caches the training dataset using multi-threaded worker pool
# Balance between performance and stability, avoids Ray-related issues

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
echo "Using single machine thread pool worker"

# Create cache directory
mkdir -p "$NAVSIM_EXP_ROOT/training_cache"

# Run caching command with thread pool worker
python $NAVSIM_DEVKIT_ROOT/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_style_agent \
    experiment_name=training_diffusiondrive_style_agent \
    train_test_split=styletrain \
    cache_path=$NAVSIM_EXP_ROOT/training_cache \
    worker=single_machine_thread_pool 