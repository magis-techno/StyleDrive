#!/bin/bash

echo "=== Cleaning up Ray processes ==="

# 检查是否有Ray进程运行
echo "Checking for existing Ray processes..."
ray_processes=$(ps aux | grep -E "(ray|python.*ray)" | grep -v grep | wc -l)

if [ "$ray_processes" -gt 0 ]; then
    echo "Found $ray_processes Ray-related processes:"
    ps aux | grep -E "(ray|python.*ray)" | grep -v grep
    echo ""
    
    # 尝试优雅关闭Ray
    echo "Attempting graceful Ray shutdown..."
    python -c "import ray; ray.shutdown()" 2>/dev/null || echo "Ray not initialized in Python"
    
    # 强制杀死Ray进程
    echo "Force killing Ray processes..."
    pkill -f ray 2>/dev/null || echo "No Ray processes to kill"
    
    sleep 2
    
    # 检查是否清理干净
    remaining=$(ps aux | grep -E "(ray|python.*ray)" | grep -v grep | wc -l)
    if [ "$remaining" -eq 0 ]; then
        echo "✓ All Ray processes cleaned up"
    else
        echo "⚠ Some Ray processes still running:"
        ps aux | grep -E "(ray|python.*ray)" | grep -v grep
    fi
else
    echo "✓ No Ray processes found"
fi

echo ""
echo "=== Cleanup complete ===" 