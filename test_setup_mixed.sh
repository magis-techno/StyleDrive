#!/bin/bash

# StyleDrive Setup Test Script (Mixed Configuration)
# Data in DiffusionDrive directory, code in StyleDrive directory

echo "=================================================="
echo "StyleDrive Setup Test - Mixed Configuration"
echo "=================================================="
echo ""

# 1. Set environment variables

echo "   Data paths (DiffusionDrive):"
echo "     NUPLAN_MAPS_ROOT: $NUPLAN_MAPS_ROOT"
echo "     OPENSCENE_DATA_ROOT: $OPENSCENE_DATA_ROOT"
echo "   Project paths (StyleDrive):"
echo "     NAVSIM_EXP_ROOT: $NAVSIM_EXP_ROOT"
echo "     NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT"
echo ""

# 2. Check directory structure
echo "2. Checking directory structure..."

# Check DiffusionDrive data directories
DIFFUSION_DATASET="/mnt/sdb/DiffusionDrive/dataset"
if [ -d "$DIFFUSION_DATASET" ]; then
    echo "   ✓ DiffusionDrive dataset exists: $DIFFUSION_DATASET"
else
    echo "   ✗ DiffusionDrive dataset NOT found: $DIFFUSION_DATASET"
    exit 1
fi

if [ -d "$DIFFUSION_DATASET/extra_data" ]; then
    echo "   ✓ Extra data directory exists: $DIFFUSION_DATASET/extra_data"
else
    echo "   ✗ Extra data directory NOT found: $DIFFUSION_DATASET/extra_data"
    exit 1
fi

# Check StyleDrive project directories
if [ -d "$NAVSIM_DEVKIT_ROOT" ]; then
    echo "   ✓ StyleDrive navsim exists: $NAVSIM_DEVKIT_ROOT"
else
    echo "   ✗ StyleDrive navsim NOT found: $NAVSIM_DEVKIT_ROOT"
    exit 1
fi

echo ""

# 3. Check JSON files (in DiffusionDrive)
echo "3. Checking StyleDrive JSON files..."
STYLETRAIN_JSON="$DIFFUSION_DATASET/extra_data/styletrain.json"
STYLETEST_JSON="$DIFFUSION_DATASET/extra_data/styletest.json"

if [ -f "$STYLETRAIN_JSON" ]; then
    echo "   ✓ styletrain.json exists: $STYLETRAIN_JSON"
    file_size=$(stat --format="%s" "$STYLETRAIN_JSON")
    if [ "$file_size" -gt 100 ]; then
        echo "     File size: $(($file_size / 1024))KB"
    else
        echo "     ⚠ File seems too small: ${file_size}B"
    fi
else
    echo "   ✗ styletrain.json NOT found: $STYLETRAIN_JSON"
fi

if [ -f "$STYLETEST_JSON" ]; then
    echo "   ✓ styletest.json exists: $STYLETEST_JSON"
    file_size=$(stat --format="%s" "$STYLETEST_JSON")
    if [ "$file_size" -gt 100 ]; then
        echo "     File size: $(($file_size / 1024))KB"
    else
        echo "     ⚠ File seems too small: ${file_size}B"
    fi
else
    echo "   ✗ styletest.json NOT found: $STYLETEST_JSON"
fi

echo ""

# 4. Check configuration files (in StyleDrive)
echo "4. Checking configuration files..."
CONFIG_FILES=(
    "$NAVSIM_DEVKIT_ROOT/planning/script/config/common/agent/diffusiondrive_style_agent.yaml"
    "$NAVSIM_DEVKIT_ROOT/planning/script/config/common/agent/transfuser_style_agent.yaml"
    "$NAVSIM_DEVKIT_ROOT/planning/script/config/common/agent/ego_status_mlp_style_agent.yaml"
)

for config_file in "${CONFIG_FILES[@]}"; do
    if [ -f "$config_file" ]; then
        echo "   ✓ Config exists: $(basename $config_file)"
        # Check if it still has placeholder paths
        if grep -q "YourStyleDrivePath" "$config_file"; then
            echo "     ⚠ Contains placeholder path - needs updating"
        else
            echo "     ✓ Path configuration looks good"
        fi
    else
        echo "   ✗ Config NOT found: $(basename $config_file)"
    fi
done

echo ""

# 5. Test JSON file validity
echo "5. Testing JSON file validity..."
if [ -f "$STYLETRAIN_JSON" ]; then
    if python3 -c "import json; json.load(open('$STYLETRAIN_JSON'))" 2>/dev/null; then
        echo "   ✓ styletrain.json is valid JSON"
    else
        echo "   ✗ styletrain.json is INVALID JSON"
    fi
fi

if [ -f "$STYLETEST_JSON" ]; then
    if python3 -c "import json; json.load(open('$STYLETEST_JSON'))" 2>/dev/null; then
        echo "   ✓ styletest.json is valid JSON"
    else
        echo "   ✗ styletest.json is INVALID JSON"
    fi
fi

echo ""

# 6. Create necessary directories
echo "6. Creating necessary directories..."
mkdir -p "$NAVSIM_EXP_ROOT/training_cache"
if [ -d "$NAVSIM_EXP_ROOT/training_cache" ]; then
    echo "   ✓ Cache directory created: $NAVSIM_EXP_ROOT/training_cache"
else
    echo "   ✗ Failed to create cache directory"
fi

echo ""

# 7. System resources check
echo "7. System resources check..."
echo "   CPU cores: $(nproc)"
echo "   Available memory: $(free -h | awk '/^Mem:/{print $7}')"
echo "   Disk space (DiffusionDrive data): $(df -h $DIFFUSION_DATASET | tail -1 | awk '{print $4}') available"
echo "   Disk space (StyleDrive exp): $(df -h $NAVSIM_EXP_ROOT | tail -1 | awk '{print $4}') available"

echo ""
echo "=================================================="
echo "Setup test completed!"
echo "=================================================="

# Summary
echo ""
echo "Configuration Summary:"
echo "  Data source: DiffusionDrive directory (reused)"
echo "  Code base: StyleDrive directory"
echo "  Cache target: StyleDrive/exp directory"
echo ""
echo "Next steps:"
echo "1. If JSON files are missing, download them to DiffusionDrive/dataset/extra_data/"
echo "2. If config files have placeholder paths, run the config update script"
echo "3. Run the dataset caching script"
echo "" 