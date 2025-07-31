#!/bin/bash

# StyleDrive Environment Variables (Mixed Configuration)
# Data reuses DiffusionDrive directory, code in separate StyleDrive directory
# Usage: source ./env_vars_mixed.sh

# Data-related paths (reuse DiffusionDrive)
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/sdb/DiffusionDrive/dataset/maps"
export OPENSCENE_DATA_ROOT="/mnt/sdb/DiffusionDrive/dataset"

# StyleDrive project paths
export NAVSIM_EXP_ROOT="/mnt/sdb/StyleDrive/exp"
export NAVSIM_DEVKIT_ROOT="/mnt/sdb/StyleDrive/navsim"

echo "Mixed configuration environment variables set:"
echo "  Data paths (DiffusionDrive):"
echo "    NUPLAN_MAPS_ROOT=$NUPLAN_MAPS_ROOT"
echo "    OPENSCENE_DATA_ROOT=$OPENSCENE_DATA_ROOT"
echo "  Project paths (StyleDrive):"
echo "    NAVSIM_EXP_ROOT=$NAVSIM_EXP_ROOT"
echo "    NAVSIM_DEVKIT_ROOT=$NAVSIM_DEVKIT_ROOT" 