#!/bin/bash

# Script to run multi-style comparison for a single scene
# Usage: ./run_multi_style_comparison.sh [CHECKPOINT_PATH] [SCENE_TOKEN]

CKPT=${1:-YOUR_CKPT_PATH}
SCENE_TOKEN=${2}
AGENT_TYPE=diffusiondrive_style_agent

if [ -z "$SCENE_TOKEN" ]; then
    echo "Usage: $0 <checkpoint_path> <scene_token>"
    echo "Example: $0 /path/to/ckpt.ckpt scene_12345abc"
    exit 1
fi

echo "Running multi-style comparison..."
echo "Checkpoint: $CKPT"
echo "Scene Token: $SCENE_TOKEN"
echo "Agent: $AGENT_TYPE"

python -c "
import sys
sys.path.append('.')

from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.visualization.multi_style_viz import run_multi_style_evaluation, save_multi_style_visualization
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path
import os

# Load configuration (simplified)
scene_token = '$SCENE_TOKEN'
checkpoint_path = '$CKPT'

# Setup scene loader
scene_loader = SceneLoader(
    sensor_blobs_path=None,
    data_path=Path(os.environ.get('OPENSCENE_DATA_ROOT', '/data') + '/navsim_logs/styletest'),
    scene_filter=None,
    sensor_config=SensorConfig.build_no_sensors(),
)

# Load agent (you may need to adjust this based on your agent configuration)
from navsim.agents.diffusiondrive.transfuser_agent import DiffusionDriveAgent
agent = DiffusionDriveAgent(checkpoint_path=checkpoint_path)
agent.initialize()

try:
    # Get scene and agent input
    scene = scene_loader.get_scene_from_token(scene_token)
    agent_input = scene_loader.get_agent_input_from_token(scene_token)
    
    # Run multi-style evaluation
    print('Running evaluation for multiple styles...')
    results = run_multi_style_evaluation(
        agent_input=agent_input,
        scene=scene,
        agent=agent,
        styles=['aggressive', 'normal', 'conservative']
    )
    
    # Save visualization
    output_dir = './multi_style_results'
    viz_path = save_multi_style_visualization(results, output_dir, scene_token)
    
    print(f'Multi-style comparison saved to: {viz_path}')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
"

echo "Multi-style comparison completed."