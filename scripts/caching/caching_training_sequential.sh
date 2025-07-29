#!/bin/bash

# 使用sequential worker，完全避免Ray相关问题
# 这是最稳定但最慢的方案

# 确保环境变量已设置
if [ -z "$NAVSIM_DEVKIT_ROOT" ]; then
    echo "Error: NAVSIM_DEVKIT_ROOT is not set. Please run: source ./env_vars.sh"
    exit 1
fi

if [ -z "$NAVSIM_EXP_ROOT" ]; then
    echo "Error: NAVSIM_EXP_ROOT is not set. Please run: source ./env_vars.sh"
    exit 1
fi

echo "Using NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT"
echo "Using NAVSIM_EXP_ROOT: $NAVSIM_EXP_ROOT"
echo "Using sequential worker (single-threaded, most stable)"

# 创建缓存目录
mkdir -p "$NAVSIM_EXP_ROOT/training_cache"

# 运行缓存命令，使用sequential worker
python $NAVSIM_DEVKIT_ROOT/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_style_agent \
    experiment_name=training_diffusiondrive_style_agent \
    train_test_split=styletrain \
    cache_path=$NAVSIM_EXP_ROOT/training_cache \
    worker=sequential 