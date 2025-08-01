# StyleTrajectoryApp 命令行使用指南

## 🚀 快速开始

### 基础用法
```bash
# 方式1: 通过模块运行 (推荐) - 简单视图
python -m style_trajectory_app.cli --checkpoint /path/to/model.ckpt --split navtest

# 方式2: 直接运行脚本 - BEV视图
python style_trajectory_app/cli.py --checkpoint /path/to/model.ckpt --split navmini --view-type bev

# 方式3: 使用便捷脚本 - 高级可视化
python run_style_demo.py --checkpoint /path/to/model.ckpt --split styletrain --view-type bev
```

### 完整参数示例
```bash
# 简单轨迹可视化
python -m style_trajectory_app.cli \
  --checkpoint /path/to/diffusiondrive_style.ckpt \
  --split navtest \
  --output ./results \
  --scenes 5 \
  --view-type simple \
  --seed 42 \
  --verbose

# BEV轨迹可视化（推荐）
python -m style_trajectory_app.cli \
  --checkpoint /path/to/diffusiondrive_style.ckpt \
  --split navtest \
  --output ./results \
  --scenes 3 \
  --view-type bev \
  --seed 42 \
  --verbose
```

## 📝 命令行参数

| 参数 | 短名 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| `--checkpoint` | `-c` | str | ✅ | - | DiffusionDrive-Style模型检查点路径 |
| `--split` | `-sp` | str | ❌ | `navtest` | 数据集split名称 (navtest/navmini/styletrain等) |
| `--output` | `-o` | str | ❌ | `./style_trajectory_results` | 输出目录 |
| `--scenes` | - | int | ❌ | `1` | 要处理的场景数量 |
| `--view-type` | - | str | ❌ | `simple` | 可视化类型 (simple/bev) |
| `--lr` | - | float | ❌ | `6e-4` | 学习率 |
| `--seed` | - | int | ❌ | `42` | 随机种子 |
| `--verbose` | `-v` | flag | ❌ | `False` | 详细输出模式 |

## 🎨 可视化类型

### Simple View (简单视图)
- **特点**: 简洁的2D轨迹对比图
- **优势**: 快速生成，资源占用少
- **适用**: 快速验证和调试

### BEV View (鸟瞰图视图) 🌟 推荐
- **特点**: 包含地图背景和车辆标注的专业BEV视图
- **优势**: 更直观理解轨迹与环境关系
- **包含**: 道路结构、车辆位置、轨迹对比
- **适用**: 正式分析和展示

```bash
# 使用BEV视图 (推荐)
python -m style_trajectory_app.cli -c model.ckpt --split navtest --view-type bev

# 使用简单视图 (快速测试)
python -m style_trajectory_app.cli -c model.ckpt --split navtest --view-type simple
```

## 📁 输出结构

运行后会在输出目录生成以下文件：

```
style_trajectory_results/
├── visualizations/           # 可视化图片
│   ├── scene_001_a1b2c3d4.png
│   ├── scene_002_e5f6g7h8.png
│   └── ...
├── data/                    # 轨迹数据
│   ├── scene_001_a1b2c3d4_data.json
│   ├── scene_002_e5f6g7h8_data.json
│   └── ...
├── logs/                    # 日志文件 (预留)
└── summary.json             # 汇总报告
```

## 📊 输出文件说明

### 可视化图片 (`visualizations/*.png`)
- 三种风格的轨迹对比图
- 高分辨率 (300 DPI)
- PNG格式，适合论文和报告

### 轨迹数据 (`data/*_data.json`)
每个场景的详细数据，包含：
```json
{
  "scene_token": "场景唯一标识",
  "scene_metadata": {
    "map_name": "地图名称",
    "log_name": "日志名称",
    "camera_frames": 10,
    "lidar_frames": 10
  },
  "trajectories": {
    "aggressive": {
      "trajectory": [[x1,y1,heading1], ...],
      "start_point": [x, y],
      "end_point": [x, y],
      "total_distance": 25.6
    },
    "normal": { ... },
    "conservative": { ... }
  },
  "timing": {
    "prediction_time": 2.34,
    "total_demo_time": 3.12
  }
}
```

### 汇总报告 (`summary.json`)
包含所有场景的统计信息：
```json
{
  "total_scenes": 5,
  "successful_scenes": 5,
  "failed_scenes": 0,
  "total_time": 15.67,
  "average_time_per_scene": 3.13,
  "config": { ... },
  "results": [ ... ]
}
```

## 💡 使用示例

### 单个场景快速测试
```bash
# 简单视图测试
python -m style_trajectory_app.cli -c model.ckpt --split navtest -o ./test_run

# BEV视图测试  
python -m style_trajectory_app.cli -c model.ckpt --split navtest -o ./test_run --view-type bev
```

### 批量处理多个场景
```bash
# 简单轨迹批量处理
python -m style_trajectory_app.cli \
  -c /models/diffusiondrive_style.ckpt \
  --split navmini \
  -o ./batch_results \
  --scenes 10 \
  --verbose

# BEV轨迹批量处理（推荐）
python -m style_trajectory_app.cli \
  -c /models/diffusiondrive_style.ckpt \
  --split navmini \
  -o ./batch_results \
  --scenes 10 \
  --view-type bev \
  --verbose
```

### 可重现实验
```bash
# BEV视图大规模实验
python -m style_trajectory_app.cli \
  -c model.ckpt \
  --split styletrain \
  -o ./experiment_1 \
  --scenes 20 \
  --view-type bev \
  --seed 12345
```

## 🔧 故障排除

### 常见错误
1. **模型文件不存在**
   ```
   ❌ 检查点文件不存在: /path/to/model.ckpt
   ```
   解决：检查模型路径是否正确

2. **环境变量未设置**
   ```
   ❌ 环境变量 OPENSCENE_DATA_ROOT 未设置
   ```
   解决：设置环境变量指向数据集根目录
   ```bash
   export OPENSCENE_DATA_ROOT=/path/to/your/data/root
   ```

3. **GPU内存不足**
   ```
   CUDA out of memory
   ```
   解决：减少batch size或使用CPU推理

### 调试技巧
- 使用 `--verbose` 查看详细输出
- 先用 `--scenes 1` 测试单个场景
- 检查输出目录的权限设置
- 确认环境变量设置正确

### 环境变量设置
```bash
# 临时设置 (当前会话有效)
export OPENSCENE_DATA_ROOT=/path/to/your/data/root

# 永久设置 (添加到 ~/.bashrc 或 ~/.zshrc)
echo 'export OPENSCENE_DATA_ROOT=/path/to/your/data/root' >> ~/.bashrc
source ~/.bashrc
```

## 🎯 性能优化

### 推荐配置
- **GPU**: 8GB+ 显存
- **内存**: 16GB+ 系统内存  
- **存储**: SSD 推荐 (数据集访问)

### 批量处理建议
```bash
# 大批量处理 (50+ 场景) - BEV视图
python -m style_trajectory_app.cli \
  -c model.ckpt \
  --split navtest \
  -o ./large_batch \
  --scenes 100 \
  --view-type bev \
  --seed 42 \
  > large_batch.log 2>&1 &
```