#!/bin/bash

# Example script to run PDM scoring with visualization
# Usage: ./run_pdm_score_with_visualization.sh [CHECKPOINT_PATH] [AGENT_TYPE]

TRAIN_TEST_SPLIT=styletest
CKPT=${1:-YOUR_CKPT_PATH}
AGENT_TYPE=${2:-diffusiondrive_style_agent}

echo "Running PDM evaluation with visualization..."
echo "Checkpoint: $CKPT"
echo "Agent: $AGENT_TYPE"
echo "Split: $TRAIN_TEST_SPLIT"

python $NAVSIM_DEVKIT_ROOT/planning/script/run_pdm_score_with_visualization.py \
    --config-path="config/pdm_scoring" \
    --config-name="default_run_pdm_score_with_visualization" \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=$AGENT_TYPE \
    agent.checkpoint_path=$CKPT \
    experiment_name=eval_${AGENT_TYPE}_with_viz \
    enable_visualization=true \
    max_visualizations=20

echo "Evaluation completed. Check output directory for visualizations."