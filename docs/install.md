# Download and installation

### 1. Clone the navsim-devkit

Clone the repository

```bash
git clone https://github.com/AIR-THU/StyleDrive.git
cd StyleDrive
```

### 2. Download StyleDrive dataset

Download styletrain.json and styletest.json from huggingface.
```bash
mkdir dataset/extra_data
cd dataset/extra_data
wget https://huggingface.co/datasets/Ryhn98/StyleDrive-Dataset/blob/main/styletest.json
wget https://huggingface.co/datasets/Ryhn98/StyleDrive-Dataset/blob/main/styletrain.json
cd ../..
```

Then, you need to download the OpenScene logs and sensor blobs, as well as the nuPlan maps.
We provide scripts to download the nuplan maps, the mini split and the test split.
Navigate to the download directory and download the maps

```bash
cd download && ./download_maps
```

Next download the following data splits:

```bash
./download_navtrain
./download_test
```

This will download the splits into the download directory. From there, move it to create the following structure.

```angular2html
~/StyleDrive
├── navsim (containing the devkit)
├── exp
└── dataset
    ├── extra_data
    |    ├── styletrain.json
    |    ├── styletest.json
    ├── maps
    ├── navsim_logs
    |    ├── test
    |    ├── trainval
    └── sensor_blobs
    |    ├── test
    |    ├── trainval
```
Set the required environment variables, by adding the following to your `~/.bashrc` file
Based on the structure above, the environment variables need to be defined as:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/StyleDrive/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/StyleDrive/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/StyleDrive/navsim"
export OPENSCENE_DATA_ROOT="$HOME/StyleDrive/dataset"
```

### 3. Install the navsim-devkit and specific packages for baseline models

Then, install navsim.
To this end, create a new environment and install the required dependencies:

```bash
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .
pip install diffusers einops 
```

### 4. Modify Path in Agents' Configs

Finally, modify the styletrain_path and styletest_path in Agents' configs:
1. $NAVSIM_DEVKIT_ROOT/planning/script/config/common/agent/transfuser_style_agent.yaml
2. $NAVSIM_DEVKIT_ROOT/planning/script/config/common/agent/diffusiondrive_style_agent.yaml
3. $NAVSIM_DEVKIT_ROOT/planning/script/config/common/agent/ego_status_mlp_style_agent.yaml

### 5. Set Environment Variables

Create environment variable setup script for easy use:

```bash
# Set environment variables
source ./env_vars.sh
```

Or manually set them:
```bash
export NAVSIM_EXP_ROOT="/path/to/your/StyleDrive/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/your/StyleDrive/navsim"
```

### 6. Dataset Caching

Before training, you need to cache the dataset. Due to Ray compatibility issues on some systems, we provide multiple caching options:

#### Option 1: Sequential Worker (Most Stable)
```bash
bash scripts/caching/caching_training_sequential.sh
```
- **Pros**: Most stable, works on all systems
- **Cons**: Single-threaded, slower

#### Option 2: Thread Pool Worker (Recommended)
```bash
bash scripts/caching/caching_training_threadpool.sh
```
- **Pros**: Multi-threaded, faster than sequential
- **Cons**: May have compatibility issues on some systems

#### Option 3: Ray Worker (Fastest, if working)
```bash
bash scripts/caching/caching_training_tran_diff.sh
```
- **Pros**: Fastest parallel processing
- **Cons**: May fail due to Ray resource issues

**Note**: If you encounter "Resource temporarily unavailable" errors with Ray, use Option 1 or 2. 

For detailed troubleshooting, see [troubleshooting.md](troubleshooting.md).