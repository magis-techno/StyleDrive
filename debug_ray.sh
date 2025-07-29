#!/bin/bash

echo "=== Ray Debug Information ==="
echo ""

# 检查Ray版本
echo "1. Ray Version:"
python -c "import ray; print(f'Ray version: {ray.__version__}')" 2>/dev/null || echo "Ray not installed or import failed"

echo ""

# 检查网络配置
echo "2. Network Configuration:"
echo "Hostname: $(hostname)"
echo "IP addresses:"
ip addr show | grep "inet " | grep -v "127.0.0.1" | awk '{print "  " $2}' || echo "  Unable to get IP addresses"

echo ""

# 检查系统资源限制
echo "3. System Limits:"
echo "Max open files: $(ulimit -n)"
echo "Max processes: $(ulimit -u)"
echo "Max memory (KB): $(ulimit -m)"

echo ""

# 检查端口使用情况
echo "4. Port Usage (Ray common ports):"
for port in 6379 8000 8265 10001; do
    if netstat -ln 2>/dev/null | grep ":$port " >/dev/null; then
        echo "  Port $port: IN USE"
    else
        echo "  Port $port: available"
    fi
done

echo ""

# 尝试简单的Ray初始化
echo "5. Ray Initialization Test:"
python -c "
import ray
try:
    ray.init(num_cpus=1, ignore_reinit_error=True)
    print('✓ Ray initialization successful')
    ray.shutdown()
except Exception as e:
    print(f'✗ Ray initialization failed: {e}')
" 2>&1

echo ""
echo "=== Debug Complete ===" 