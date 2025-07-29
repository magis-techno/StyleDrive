#!/bin/bash

# StyleDrive Agent Configuration Path Setup Script
# This script updates the style data paths in all style agent configurations

# Ensure environment variables are set
if [ -z "$NAVSIM_DEVKIT_ROOT" ]; then
    echo "Error: NAVSIM_DEVKIT_ROOT is not set. Please run: source ./env_vars.sh"
    exit 1
fi

# Get the StyleDrive root directory (parent of navsim)
STYLEDRIVE_ROOT=$(dirname "$NAVSIM_DEVKIT_ROOT")
STYLE_DATA_PATH="$STYLEDRIVE_ROOT/dataset/extra_data"

echo "=== StyleDrive Agent Path Setup ==="
echo "StyleDrive root: $STYLEDRIVE_ROOT"
echo "Style data path: $STYLE_DATA_PATH"
echo ""

# Check if style data files exist
if [ ! -f "$STYLE_DATA_PATH/styletrain.json" ]; then
    echo "⚠️  Warning: styletrain.json not found at $STYLE_DATA_PATH/styletrain.json"
    echo "    Please download the style data files as per installation instructions"
fi

if [ ! -f "$STYLE_DATA_PATH/styletest.json" ]; then
    echo "⚠️  Warning: styletest.json not found at $STYLE_DATA_PATH/styletest.json"
    echo "    Please download the style data files as per installation instructions"
fi

echo ""

# List of agent config files to update
AGENT_CONFIGS=(
    "$NAVSIM_DEVKIT_ROOT/planning/script/config/common/agent/diffusiondrive_style_agent.yaml"
    "$NAVSIM_DEVKIT_ROOT/planning/script/config/common/agent/transfuser_style_agent.yaml"
    "$NAVSIM_DEVKIT_ROOT/planning/script/config/common/agent/ego_status_mlp_style_agent.yaml"
)

# Update each config file
for config_file in "${AGENT_CONFIGS[@]}"; do
    if [ -f "$config_file" ]; then
        echo "Updating: $(basename "$config_file")"
        
        # Create backup
        cp "$config_file" "$config_file.backup"
        
        # Update paths using sed
        sed -i "s|styletrain_path:.*|styletrain_path: \"$STYLE_DATA_PATH/styletrain.json\"|g" "$config_file"
        sed -i "s|styletest_path:.*|styletest_path: \"$STYLE_DATA_PATH/styletest.json\"|g" "$config_file"
        
        echo "  ✓ Updated style data paths"
    else
        echo "  ✗ Config file not found: $config_file"
    fi
done

echo ""
echo "=== Setup Complete ==="
echo "All style agent configurations have been updated with correct paths."
echo "Backups of original files are saved with .backup extension." 