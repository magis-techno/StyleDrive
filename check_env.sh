#!/bin/bash

echo "=== StyleDrive Environment Check ==="
echo ""

# 检查环境变量
echo "1. Environment Variables:"
if [ -n "$NAVSIM_DEVKIT_ROOT" ]; then
    echo "   ✓ NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT"
else
    echo "   ✗ NAVSIM_DEVKIT_ROOT: NOT SET"
fi

if [ -n "$NAVSIM_EXP_ROOT" ]; then
    echo "   ✓ NAVSIM_EXP_ROOT: $NAVSIM_EXP_ROOT"
else
    echo "   ✗ NAVSIM_EXP_ROOT: NOT SET"
fi

echo ""

# 检查路径存在性
echo "2. Path Existence:"
if [ -d "$NAVSIM_DEVKIT_ROOT" ]; then
    echo "   ✓ NAVSIM_DEVKIT_ROOT directory exists"
else
    echo "   ✗ NAVSIM_DEVKIT_ROOT directory NOT found"
fi

if [ -d "$NAVSIM_EXP_ROOT" ]; then
    echo "   ✓ NAVSIM_EXP_ROOT directory exists"
else
    echo "   ✗ NAVSIM_EXP_ROOT directory NOT found - will be created if needed"
fi

echo ""

# 检查关键文件
echo "3. Key Files:"
CONFIG_FILE="$NAVSIM_DEVKIT_ROOT/planning/script/config/training/default_training.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "   ✓ Training config file exists"
else
    echo "   ✗ Training config file NOT found: $CONFIG_FILE"
fi

WORKER_CONFIG="$NAVSIM_DEVKIT_ROOT/planning/script/config/common/worker/ray_low_resource.yaml"
if [ -f "$WORKER_CONFIG" ]; then
    echo "   ✓ Low resource worker config exists"
else
    echo "   ✗ Low resource worker config NOT found: $WORKER_CONFIG"
fi

echo ""

# 检查系统资源
echo "4. System Resources:"
echo "   CPU cores: $(nproc)"
echo "   Available memory: $(free -h | awk '/^Mem:/{print $7}')"

echo ""
echo "=== Check Complete ===" 