#!/bin/bash

# Update StyleDrive configuration files for mixed configuration
# Data in DiffusionDrive directory, code in StyleDrive directory

echo "Updating StyleDrive configuration files for mixed configuration..."
echo ""

# Set the correct paths for mixed configuration
DIFFUSION_DATASET="$HOME/DiffusionDrive/dataset"
NAVSIM_DEVKIT_ROOT="$HOME/StyleDrive/navsim"

echo "Configuration:"
echo "  Data source: $DIFFUSION_DATASET"
echo "  Config location: $NAVSIM_DEVKIT_ROOT"
echo ""

# Configuration files to update
CONFIG_FILES=(
    "$NAVSIM_DEVKIT_ROOT/planning/script/config/common/agent/diffusiondrive_style_agent.yaml"
    "$NAVSIM_DEVKIT_ROOT/planning/script/config/common/agent/transfuser_style_agent.yaml"
    "$NAVSIM_DEVKIT_ROOT/planning/script/config/common/agent/ego_status_mlp_style_agent.yaml"
)

# Update each configuration file
for config_file in "${CONFIG_FILES[@]}"; do
    if [ -f "$config_file" ]; then
        echo "Updating: $(basename $config_file)"
        
        # Create backup
        cp "$config_file" "$config_file.backup"
        
        # Update paths to point to DiffusionDrive dataset
        sed -i "s|YourStyleDrivePath|$DIFFUSION_DATASET|g" "$config_file"
        
        echo "  ✓ Updated paths in $(basename $config_file)"
        echo "  ✓ Backup created: $(basename $config_file).backup"
    else
        echo "  ✗ File not found: $config_file"
    fi
done

echo ""
echo "Configuration update completed!"
echo ""
echo "Updated paths:"
echo "  styletrain_path: $DIFFUSION_DATASET/extra_data/styletrain.json"
echo "  styletest_path: $DIFFUSION_DATASET/extra_data/styletest.json"
echo ""
echo "Note: Data will be read from DiffusionDrive directory"
echo "      Cache will be stored in StyleDrive/exp directory"
echo "" 