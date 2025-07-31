#!/bin/bash

# Complete StyleDrive Setup Script (Mixed Configuration)
# Data reuses DiffusionDrive directory, code in StyleDrive directory

echo "##################################################"
echo "#        StyleDrive Complete Setup               #"
echo "#        Mixed Configuration                     #"
echo "#        Data: DiffusionDrive | Code: StyleDrive #"
echo "##################################################"
echo ""

# Step 1: Test current setup
echo "Step 1: Testing current setup..."
echo "----------------------------------------------"
chmod +x test_setup_mixed.sh
./test_setup_mixed.sh

read -p "Press Enter to continue to configuration update..."
echo ""

# Step 2: Update configurations
echo "Step 2: Updating configuration files..."
echo "----------------------------------------------"
chmod +x update_configs_mixed.sh
./update_configs_mixed.sh

read -p "Press Enter to continue to caching..."
echo ""

# Step 3: Run dataset caching
echo "Step 3: Running dataset caching..."
echo "----------------------------------------------"
chmod +x run_caching_mixed.sh
./run_caching_mixed.sh

echo ""
echo "##################################################"
echo "#              Setup Complete!                  #"
echo "##################################################"
echo ""
echo "Configuration Summary:"
echo "  ✓ Data source: DiffusionDrive/dataset (reused)"
echo "  ✓ Code base: StyleDrive/navsim"
echo "  ✓ Cache location: StyleDrive/exp/training_cache"
echo "  ✓ Environment variables configured"
echo ""
echo "Next steps:"
echo "1. Verify cache files were created in StyleDrive/exp/training_cache"
echo "2. You can now run training scripts from StyleDrive directory"
echo "3. Use 'source ./env_vars_mixed.sh' to set environment variables"
echo ""
echo "Environment variables for future use:"
echo "  export OPENSCENE_DATA_ROOT=\"\$HOME/DiffusionDrive/dataset\""
echo "  export NAVSIM_DEVKIT_ROOT=\"\$HOME/StyleDrive/navsim\""
echo "  export NAVSIM_EXP_ROOT=\"\$HOME/StyleDrive/exp\""
echo "" 