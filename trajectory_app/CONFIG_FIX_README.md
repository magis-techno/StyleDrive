# 🔧 配置问题修复指南

## ❌ 遇到的错误

```
TypeError: __init__() missing 1 required positional argument: 'config'
```

## ✅ 解决方案

`TrajectoryPredictionApp` 类需要一个配置参数。以下是正确的使用方法：

### 方法1: 使用字典配置 (推荐用于测试)

```python
from trajectory_app.app import TrajectoryPredictionApp
import os

# 创建最小配置
config = {
    "model": {
        "type": "diffusiondrive",
        "checkpoint_path": None,  # 使用默认检查点
        "lr": 6e-4
    },
    "data": {
        "navsim_log_path": os.environ.get("OPENSCENE_DATA_ROOT", "/tmp") + "/navsim_logs/test",
        "sensor_blobs_path": os.environ.get("OPENSCENE_DATA_ROOT", "/tmp") + "/sensor_blobs/test",
        "cache_path": os.environ.get("NAVSIM_EXP_ROOT", "/tmp") + "/metric_cache"
    },
    "visualization": {
        "time_windows": [1.0, 3.0, 6.0],
        "save_formats": ["png"]
    }
}

# 现在可以正常创建应用
app = TrajectoryPredictionApp(config)
```

### 方法2: 使用YAML配置文件

```python
from trajectory_app.app import TrajectoryPredictionApp

# 使用默认配置文件
app = TrajectoryPredictionApp("trajectory_app/config/default_config.yaml")
```

### 方法3: 环境变量设置

如果你有完整的NavSim环境，设置以下环境变量：

```bash
export OPENSCENE_DATA_ROOT="/path/to/your/openscene/data"
export NAVSIM_EXP_ROOT="/path/to/your/navsim/experiments"
```

然后使用默认配置：

```python
app = TrajectoryPredictionApp("trajectory_app/config/default_config.yaml")
```

## 🧪 快速验证

运行验证脚本检查修复是否正确：

```bash
# 从项目根目录运行
python verify_config_fix.py
```

如果看到 `🎉 验证成功!`，说明配置修复正常工作。

## 📝 关键变更

1. **构造函数**: `TrajectoryPredictionApp()` → `TrajectoryPredictionApp(config)`
2. **自动初始化**: 构造函数会自动初始化所有组件，无需调用 `app.initialize()`
3. **配置必需**: 必须提供model、data、visualization配置段

## 🔍 故障排除

### 问题1: 环境变量未设置
```
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/navsim_logs/test'
```
**解决**: 设置正确的 `OPENSCENE_DATA_ROOT` 和 `NAVSIM_EXP_ROOT` 环境变量

### 问题2: 检查点文件未找到
```
RuntimeError: No checkpoint file found
```
**解决**: 
- 设置 `checkpoint_path` 为具体的.pth文件路径
- 或者确保默认路径下有可用的检查点

### 问题3: 数据路径不存在
```
ValueError: NavSim data path does not exist
```
**解决**: 确保 `navsim_log_path` 和 `sensor_blobs_path` 指向有效的数据目录

---

这个修复确保了所有用户都能正确初始化应用，无论是否有完整的NavSim环境设置。