#!/bin/bash

# StyleDrive Environment Variables Setup Script
# This script sets up the required environment variables for StyleDrive
# Interactive version that can optionally add variables to ~/.bashrc

echo "Setting up StyleDrive environment variables..."

# Set the environment variables
export NAVSIM_EXP_ROOT="/mnt/sdb/StyleDrive/exp"
export NAVSIM_DEVKIT_ROOT="/mnt/sdb/StyleDrive/navsim"

echo "Environment variables set:"
echo "  NAVSIM_EXP_ROOT=$NAVSIM_EXP_ROOT"
echo "  NAVSIM_DEVKIT_ROOT=$NAVSIM_DEVKIT_ROOT"

# Optionally add to ~/.bashrc for permanent effect
read -p "Do you want to add these variables to ~/.bashrc for permanent effect? (y/n): " -n 1 -r
echo    # move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "" >> ~/.bashrc
    echo "# StyleDrive environment variables" >> ~/.bashrc
    echo "export NAVSIM_EXP_ROOT=\"/mnt/sdb/StyleDrive/exp\"" >> ~/.bashrc
    echo "export NAVSIM_DEVKIT_ROOT=\"/mnt/sdb/StyleDrive/navsim\"" >> ~/.bashrc
    echo "Variables added to ~/.bashrc"
    echo "Please run 'source ~/.bashrc' or restart your terminal to make them permanent."
fi

echo "Setup complete!" 