# StyleDrive Setup - Mixed Configuration

This setup reuses data from an existing DiffusionDrive installation while maintaining StyleDrive code in a separate directory.

## Configuration Overview

```
DiffusionDrive/                    StyleDrive/
├── dataset/ (REUSED)             ├── navsim/ (StyleDrive code)
│   ├── extra_data/               ├── exp/ (experiments & cache)
│   │   ├── styletrain.json       └── scripts/
│   │   └── styletest.json        
│   ├── maps/                     
│   ├── navsim_logs/              
│   └── sensor_blobs/             
```

## Environment Variables

```bash
# Data paths (DiffusionDrive)
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/DiffusionDrive/dataset/maps"
export OPENSCENE_DATA_ROOT="$HOME/DiffusionDrive/dataset"

# Project paths (StyleDrive)
export NAVSIM_EXP_ROOT="$HOME/StyleDrive/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/StyleDrive/navsim"
```

## Quick Start

### Option 1: Complete Automated Setup (Recommended)
```bash
chmod +x complete_setup_mixed.sh
./complete_setup_mixed.sh
```

### Option 2: Step-by-Step Setup

#### 1. Test Current Setup
```bash
chmod +x test_setup_mixed.sh
./test_setup_mixed.sh
```

#### 2. Update Configuration Files
```bash
chmod +x update_configs_mixed.sh
./update_configs_mixed.sh
```

#### 3. Set Environment Variables
```bash
source ./env_vars_mixed.sh
```

#### 4. Run Dataset Caching
```bash
chmod +x run_caching_mixed.sh
./run_caching_mixed.sh
```

## Files Description

| File | Purpose |
|------|---------|
| `complete_setup_mixed.sh` | Complete automated setup |
| `test_setup_mixed.sh` | Test system and file availability |
| `update_configs_mixed.sh` | Update config files with correct paths |
| `env_vars_mixed.sh` | Set environment variables |
| `run_caching_mixed.sh` | Run dataset caching |

## Configuration Updates

The scripts will update these configuration files in StyleDrive:
- `StyleDrive/navsim/planning/script/config/common/agent/diffusiondrive_style_agent.yaml`
- `StyleDrive/navsim/planning/script/config/common/agent/transfuser_style_agent.yaml`  
- `StyleDrive/navsim/planning/script/config/common/agent/ego_status_mlp_style_agent.yaml`

**Update example:**
```yaml
# Before
styletrain_path: "YourStyleDrivePath/dataset/extra_data/styletrain.json"

# After  
styletrain_path: "/home/user/DiffusionDrive/dataset/extra_data/styletrain.json"
```

## Data Flow

```
DiffusionDrive/dataset/extra_data/*.json → StyleDrive/exp/training_cache/
                 ↑                                    ↓
            (Read data)                          (Write cache)
                 ↑                                    ↓
        StyleDrive/navsim/agents/ (Processing code)
```

## Advantages of Mixed Configuration

1. ✅ **No data duplication** - Reuse existing DiffusionDrive dataset
2. ✅ **Isolated experiments** - StyleDrive experiments don't interfere with DiffusionDrive
3. ✅ **Easy maintenance** - Update datasets in one place
4. ✅ **Flexible development** - Independent StyleDrive codebase

## Prerequisites

### DiffusionDrive Directory Must Have:
- `DiffusionDrive/dataset/extra_data/styletrain.json`
- `DiffusionDrive/dataset/extra_data/styletest.json`
- `DiffusionDrive/dataset/maps/`
- `DiffusionDrive/dataset/navsim_logs/`
- `DiffusionDrive/dataset/sensor_blobs/`

### StyleDrive Directory Must Have:
- `StyleDrive/navsim/` (StyleDrive source code)
- `StyleDrive/exp/` (will be created for experiments)

## Troubleshooting

### JSON Files Missing in DiffusionDrive
```bash
cd ~/DiffusionDrive/dataset/extra_data
wget https://huggingface.co/datasets/Ryhn98/StyleDrive-Dataset/resolve/main/styletrain.json
wget https://huggingface.co/datasets/Ryhn98/StyleDrive-Dataset/resolve/main/styletest.json
```

### Permission Issues
```bash
chmod +x *.sh
```

### Cache Directory Issues
```bash
mkdir -p ~/StyleDrive/exp/training_cache
```

## Validation Checklist

After setup, verify:
- [x] DiffusionDrive JSON files exist and are valid
- [x] StyleDrive configuration files have correct paths  
- [x] Environment variables point to correct directories
- [x] Cache directory created in StyleDrive/exp
- [x] Caching process completed successfully

## Performance Notes

- Uses sequential worker for maximum stability
- Data reading from DiffusionDrive, cache writing to StyleDrive
- Monitor disk space in both directories during caching
- Cache files in StyleDrive/exp can be safely deleted and regenerated 