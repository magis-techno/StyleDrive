#!/bin/bash

# Convenient script to run PDM scoring with visualization
# Usage: ./run_pdm_score_with_viz.sh [CHECKPOINT_PATH] [AGENT_TYPE] [MAX_VIZ]

TRAIN_TEST_SPLIT=styletest
CKPT=${1:-YOUR_CKPT_PATH}
AGENT_TYPE=${2:-diffusiondrive_style_agent}
MAX_VIZ=${3:-20}

echo "Running PDM evaluation with visualization..."
echo "Checkpoint: $CKPT"
echo "Agent: $AGENT_TYPE"
echo "Split: $TRAIN_TEST_SPLIT"
echo "Max visualizations: $MAX_VIZ"

python $NAVSIM_DEVKIT_ROOT/planning/script/run_pdm_score.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=$AGENT_TYPE \
    agent.checkpoint_path=$CKPT \
    experiment_name=eval_${AGENT_TYPE}_with_viz \
    enable_visualization=true \
    max_visualizations=$MAX_VIZ

echo "Evaluation completed. Check output directory for visualizations."