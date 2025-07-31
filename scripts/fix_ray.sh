#!/bin/bash

# Ray修复脚本 - 自动诊断和修复Ray相关问题
# Usage: ./scripts/fix_ray.sh

echo "🔧 StyleDrive Ray修复工具"
echo "=========================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查操作系统
check_os() {
    log_info "检查操作系统..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        log_warning "检测到Windows系统，Ray在Windows上可能不稳定"
        echo "建议使用WSL2或Linux虚拟机"
        return 1
    else
        log_success "Linux/Unix系统，适合运行Ray"
        return 0
    fi
}

# 1. 停止所有Ray进程
stop_ray() {
    log_info "停止现有Ray进程..."
    
    # 尝试优雅停止
    if command -v ray &> /dev/null; then
        ray stop --force 2>/dev/null
        log_success "Ray进程已停止"
    else
        log_warning "Ray命令未找到，可能未安装"
    fi
    
    # 强制终止Ray进程
    log_info "检查残留Ray进程..."
    if command -v pkill &> /dev/null; then
        pkill -f "ray::" 2>/dev/null || true
        pkill -f "raylet" 2>/dev/null || true
        pkill -f "ray_" 2>/dev/null || true
        log_success "清理完成"
    fi
    
    sleep 2
}

# 2. 清理Ray缓存和临时文件
clean_ray_cache() {
    log_info "清理Ray缓存和临时文件..."
    
    # 清理用户目录下的Ray文件
    if [ -d "$HOME/.ray" ]; then
        rm -rf "$HOME/.ray"
        log_success "清理 ~/.ray"
    fi
    
    # 清理临时目录下的Ray文件
    if [ -d "/tmp/ray" ]; then
        rm -rf /tmp/ray*
        log_success "清理 /tmp/ray*"
    fi
    
    # Windows下的临时文件清理
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        if [ -d "/c/Users/$USER/AppData/Local/Temp/ray" ]; then
            rm -rf "/c/Users/$USER/AppData/Local/Temp/ray"*
            log_success "清理Windows临时Ray文件"
        fi
    fi
}

# 3. 检查端口占用
check_ports() {
    log_info "检查Ray默认端口占用..."
    
    local ports=(6379 10001 8265)
    for port in "${ports[@]}"; do
        if command -v netstat &> /dev/null; then
            if netstat -tulpn 2>/dev/null | grep ":$port " > /dev/null; then
                log_warning "端口 $port 被占用"
                # 尝试找到占用进程
                local pid=$(netstat -tulpn 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 | head -1)
                if [ ! -z "$pid" ] && [ "$pid" != "-" ]; then
                    log_info "占用进程PID: $pid"
                    # 可选：自动终止占用进程（谨慎使用）
                    # kill -9 $pid 2>/dev/null || true
                fi
            else
                log_success "端口 $port 可用"
            fi
        fi
    done
}

# 4. 检查系统资源
check_resources() {
    log_info "检查系统资源..."
    
    # 检查内存
    if command -v free &> /dev/null; then
        local mem_total=$(free -m | awk 'NR==2{printf "%.1f", $2/1024}')
        local mem_avail=$(free -m | awk 'NR==2{printf "%.1f", $4/1024}')
        log_info "总内存: ${mem_total}GB, 可用: ${mem_avail}GB"
        
        if (( $(echo "$mem_avail < 2.0" | bc -l) )); then
            log_warning "可用内存不足2GB，Ray可能运行不稳定"
        fi
    fi
    
    # 检查CPU核心数
    local cpu_cores=$(nproc 2>/dev/null || echo "unknown")
    log_info "CPU核心数: $cpu_cores"
}

# 5. 重新启动Ray
start_ray() {
    log_info "启动Ray集群..."
    
    if ! command -v ray &> /dev/null; then
        log_error "Ray未安装，请先安装: pip install ray"
        return 1
    fi
    
    # 尝试启动Ray
    local ray_output
    ray_output=$(ray start --head --port=6379 --object-manager-port=8076 --disable-usage-stats 2>&1)
    local ray_exit_code=$?
    
    if [ $ray_exit_code -eq 0 ]; then
        log_success "Ray集群启动成功"
        echo "$ray_output" | grep -E "(Dashboard|Local node IP)"
    else
        log_error "Ray启动失败"
        echo "$ray_output"
        return 1
    fi
    
    sleep 3
}

# 6. 验证Ray状态
verify_ray() {
    log_info "验证Ray状态..."
    
    if ray status &> /dev/null; then
        log_success "Ray集群运行正常"
        ray status
        return 0
    else
        log_error "Ray集群状态异常"
        return 1
    fi
}

# 7. 创建Ray worker和配置
create_low_resource_config() {
    log_info "创建Ray worker和配置文件..."
    
    local config_dir="navsim/planning/script/config/common/worker"
    local worker_dir="navsim/planning/utils/multithreading"
    
    if [ ! -d "$config_dir" ]; then
        log_error "配置目录不存在: $config_dir"
        return 1
    fi
    
    if [ ! -d "$worker_dir" ]; then
        log_error "Worker目录不存在: $worker_dir"
        return 1
    fi
    
    # 创建新的Ray worker类
    cat > "$worker_dir/worker_ray_existing.py" << 'EOF'
"""
Ray worker that connects to existing Ray cluster without reinitializing.
"""
import logging
from concurrent.futures import Future
from typing import Any, Iterable, List, Optional, Union

import ray
from psutil import cpu_count

from nuplan.planning.utils.multithreading.ray_execution import ray_map
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool, WorkerResources

logger = logging.getLogger(__name__)

class RayExistingCluster(WorkerPool):
    """
    Ray worker that connects to an existing Ray cluster without reinitializing.
    """

    def __init__(
        self,
        master_node_ip: Optional[str] = None,
        threads_per_node: Optional[int] = None,
        debug_mode: bool = False,
        log_to_driver: bool = True,
        logs_subdir: str = "logs",
        use_distributed: bool = False,
    ):
        """
        Initialize Ray worker for existing cluster.
        
        :param master_node_ip: Not used, kept for compatibility
        :param threads_per_node: Not used, kept for compatibility  
        :param debug_mode: If true, execute serially for debugging
        :param log_to_driver: If true, show logs from workers
        :param logs_subdir: Subdirectory for logs
        :param use_distributed: Not used, kept for compatibility
        """
        super().__init__()
        
        self._debug_mode = debug_mode
        self._log_to_driver = log_to_driver
        
        # Check if Ray is already initialized
        if not ray.is_initialized():
            logger.error("Ray is not initialized! Please start Ray first with: ray start --head")
            raise RuntimeError("Ray cluster not found. Please start Ray first.")
        
        logger.info("Connected to existing Ray cluster")
        
        # Get cluster resources
        cluster_resources = ray.cluster_resources()
        self._number_of_cpus = int(cluster_resources.get('CPU', cpu_count()))
        self._number_of_gpus = int(cluster_resources.get('GPU', 0))
        
        logger.info(f"Ray cluster resources - CPU: {self._number_of_cpus}, GPU: {self._number_of_gpus}")
        
        self._worker_resources = WorkerResources(
            number_of_nodes=1,  # Simplified assumption
            number_of_cpus_per_node=self._number_of_cpus,
            number_of_gpus_per_node=self._number_of_gpus,
        )

    @property
    def worker_resources(self) -> WorkerResources:
        """Inherited, see superclass."""
        return self._worker_resources

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool = False) -> List[Any]:
        """Inherited, see superclass."""
        del verbose  # Not used
        if self._debug_mode:
            logger.info("Running in debug mode (serial execution)")
            # Execute serially for debugging
            if len(item_lists) == 1:
                return [task.fn(arg) for arg in item_lists[0]]
            else:
                return [task.fn(*args) for args in zip(*item_lists)]
        else:
            logger.info(f"Executing {len(item_lists[0])} tasks on Ray cluster")
            return ray_map(task, *item_lists)

    def map(self, func: Any, argument_list: List[Any]) -> List[Any]:
        """Inherited, see superclass."""
        if self._debug_mode:
            logger.info("Running in debug mode (serial execution)")
            return [func(arg) for arg in argument_list]
        else:
            logger.info(f"Executing {len(argument_list)} tasks on Ray cluster")
            return ray_map(func, argument_list)

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future:
        """Inherited, see superclass."""
        if self._debug_mode:
            # Execute synchronously for debugging
            future = Future()
            try:
                result = task(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            return future
        else:
            # Submit to Ray
            ray_future = ray.remote(task).remote(*args, **kwargs)
            # Convert Ray future to standard Future
            future = Future()
            
            def _get_result():
                try:
                    result = ray.get(ray_future)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
            
            # Start background thread to get result
            import threading
            threading.Thread(target=_get_result, daemon=True).start()
            return future

    def __enter__(self):
        """Inherited, see superclass."""
        return self

    def shutdown(self) -> None:
        """
        Shutdown the worker and clear memory.
        Note: We don't shutdown Ray as it's an existing cluster.
        """
        logger.info("Worker shutdown (Ray cluster remains running)")
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Inherited, see superclass."""
        # Don't shutdown Ray as it's an existing cluster
        logger.info("Disconnecting from Ray cluster (cluster remains running)")
        self.shutdown()
EOF
    
    log_success "创建Ray worker类: $worker_dir/worker_ray_existing.py"
    
    # 创建连接现有集群的配置
    cat > "$config_dir/ray_existing_cluster.yaml" << 'EOF'
_target_: navsim.planning.utils.multithreading.worker_ray_existing.RayExistingCluster
_convert_: 'all'
master_node_ip: null          # 不需要，自动检测现有集群
threads_per_node: null        # 不需要，使用现有集群资源
debug_mode: false             # 如果为true则串行执行用于调试
log_to_driver: true           # 显示worker日志
logs_subdir: 'logs'           # 日志子目录
use_distributed: false        # 不需要，连接现有集群
EOF
    
    # 创建低资源新集群配置（备用）
    cat > "$config_dir/ray_low_resource_new.yaml" << 'EOF'
_target_: navsim.planning.utils.multithreading.worker_ray_no_torch.RayDistributedNoTorch
_convert_: 'all'
master_node_ip: null
threads_per_node: 2       # 降低线程数减少资源消耗
debug_mode: false
log_to_driver: true
logs_subdir: 'logs'
use_distributed: false
EOF
    
    log_success "创建Ray配置文件"
}

# 8. 创建Ray缓存脚本
create_ray_caching_script() {
    log_info "创建Ray缓存脚本..."
    
    # 创建连接现有集群的脚本
    cat > "scripts/caching/caching_training_ray_existing.sh" << 'EOF'
#!/bin/bash

# StyleDrive Dataset Caching Script (Ray Existing Cluster)
# This script connects to an already running Ray cluster

# Ensure environment variables are set
if [ -z "$NAVSIM_DEVKIT_ROOT" ]; then
    echo "Error: NAVSIM_DEVKIT_ROOT is not set. Please run: source ./setup_env.sh"
    exit 1
fi

if [ -z "$NAVSIM_EXP_ROOT" ]; then
    echo "Error: NAVSIM_EXP_ROOT is not set. Please run: source ./setup_env.sh"
    exit 1
fi

echo "Using NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT"
echo "Using NAVSIM_EXP_ROOT: $NAVSIM_EXP_ROOT"
echo "Using existing Ray cluster"

# Check if Ray is running
if ! ray status >/dev/null 2>&1; then
    echo "Error: Ray cluster is not running. Please start Ray first:"
    echo "  ray start --head --disable-usage-stats"
    exit 1
fi

echo "Ray cluster is running:"
ray status

# Create cache directory
mkdir -p "$NAVSIM_EXP_ROOT/training_cache"

# Run caching command with existing Ray cluster
python $NAVSIM_DEVKIT_ROOT/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_style_agent \
    experiment_name=training_diffusiondrive_style_agent \
    train_test_split=styletrain \
    cache_path=$NAVSIM_EXP_ROOT/training_cache \
    worker=ray_existing_cluster
EOF

    chmod +x "scripts/caching/caching_training_ray_existing.sh"
    log_success "创建Ray缓存脚本: scripts/caching/caching_training_ray_existing.sh"
}

# 主函数
main() {
    echo "开始Ray修复流程..."
    echo ""
    
    # 检查操作系统
    check_os
    
    # 步骤1: 停止Ray
    stop_ray
    
    # 步骤2: 清理缓存
    clean_ray_cache
    
    # 步骤3: 检查端口
    check_ports
    
    # 步骤4: 检查资源
    check_resources
    
    # 步骤5: 启动Ray
    if start_ray; then
        # 步骤6: 验证状态
        if verify_ray; then
            log_success "🎉 Ray修复成功！"
        else
            log_warning "Ray启动但状态异常，建议使用ThreadPool模式"
        fi
    else
        log_error "Ray启动失败，建议使用ThreadPool模式"
    fi
    
    # 步骤7: 创建配置
    create_low_resource_config
    
    # 步骤8: 创建脚本
    create_ray_caching_script
    
    echo ""
    echo "🔧 修复完成！使用建议："
    echo "1. Ray已启动，使用现有集群: ./scripts/caching/caching_training_ray_existing.sh"
    echo "2. 如果仍有问题，使用ThreadPool: ./scripts/caching/caching_training_threadpool.sh"
    echo "3. 检查Ray状态: ray status"
    echo "4. 停止Ray: ray stop"
}

# 运行主函数
main "$@"