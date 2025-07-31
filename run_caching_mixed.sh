#!/bin/bash

# StyleDrive Dataset Caching Script (Mixed Configuration)
# Data from DiffusionDrive directory, code in StyleDrive directory

echo "=================================================="
echo "StyleDrive Dataset Caching - Mixed Configuration"
echo "=================================================="
echo ""

# Set environment variables for mixed configuration
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/sdb/DiffusionDrive/dataset/maps"
export OPENSCENE_DATA_ROOT="/mnt/sdb/DiffusionDrive/dataset"
export NAVSIM_EXP_ROOT="/mnt/sdb/StyleDrive/exp"
export NAVSIM_DEVKIT_ROOT="/mnt/sdb/StyleDrive/navsim"

# Verify environment variables
if [ -z "$NAVSIM_DEVKIT_ROOT" ] || [ -z "$NAVSIM_EXP_ROOT" ] || [ -z "$OPENSCENE_DATA_ROOT" ]; then
    echo "Error: Environment variables not set properly"
    exit 1
fi

echo "Environment configuration:"
echo "  Data source: $OPENSCENE_DATA_ROOT"
echo "  Code base: $NAVSIM_DEVKIT_ROOT"
echo "  Cache target: $NAVSIM_EXP_ROOT"
echo ""

# Check if dataset exists (in DiffusionDrive)
if [ ! -d "$OPENSCENE_DATA_ROOT/extra_data" ]; then
    echo "Error: Dataset directory not found: $OPENSCENE_DATA_ROOT/extra_data"
    exit 1
fi

# Check if JSON files exist (in DiffusionDrive)
STYLETRAIN_JSON="$OPENSCENE_DATA_ROOT/extra_data/styletrain.json"
STYLETEST_JSON="$OPENSCENE_DATA_ROOT/extra_data/styletest.json"

if [ ! -f "$STYLETRAIN_JSON" ]; then
    echo "Error: styletrain.json not found: $STYLETRAIN_JSON"
    echo "Please download it to DiffusionDrive dataset directory"
    exit 1
fi

if [ ! -f "$STYLETEST_JSON" ]; then
    echo "Error: styletest.json not found: $STYLETEST_JSON"
    echo "Please download it to DiffusionDrive dataset directory"
    exit 1
fi

echo "JSON files verified (DiffusionDrive dataset):"
echo "  ✓ $STYLETRAIN_JSON"
echo "  ✓ $STYLETEST_JSON"
echo ""

# Create cache directory (in StyleDrive)
mkdir -p "$NAVSIM_EXP_ROOT/training_cache"
echo "Cache directory (StyleDrive): $NAVSIM_EXP_ROOT/training_cache"
echo ""

# Check if StyleDrive code exists
if [ ! -f "$NAVSIM_DEVKIT_ROOT/planning/script/run_dataset_caching.py" ]; then
    echo "Error: StyleDrive caching script not found: $NAVSIM_DEVKIT_ROOT/planning/script/run_dataset_caching.py"
    exit 1
fi

# Change to StyleDrive directory
cd "/mnt/sdb/StyleDrive"

# Run caching with sequential worker (most stable)
echo "Starting dataset caching (this may take a while)..."
echo "Using sequential worker for maximum stability"
echo "Reading data from: DiffusionDrive/dataset"
echo "Writing cache to: StyleDrive/exp/training_cache"
echo ""

python $NAVSIM_DEVKIT_ROOT/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_style_agent \
    experiment_name=training_diffusiondrive_style_agent \
    train_test_split=styletrain \
    cache_path=$NAVSIM_EXP_ROOT/training_cache \
    worker=sequential

echo ""
echo "=================================================="
echo "Caching completed!"
echo ""
echo "Data flow summary:"
echo "  JSON files: $OPENSCENE_DATA_ROOT/extra_data/ → Cache"
echo "  Cache files: $NAVSIM_EXP_ROOT/training_cache/"
echo "==================================================" 