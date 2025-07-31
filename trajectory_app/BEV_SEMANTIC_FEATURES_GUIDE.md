# 🎨 BEV语义分割特征可视化指南

## 📋 功能概述

我们已经成功实现了 **Task 1.1: BEV语义分割图可视化**！这是DiffusionDrive模型特征可视化TodoList中的第一个重要里程碑。

## ✨ 新功能特性

### 🔧 核心组件

1. **特征提取** (`inference_engine.py`)
   - 自动从模型输出中提取 `bev_semantic_map`
   - 包含预测类别、原始logits和置信度
   - 无侵入式设计，不影响原有推理流程

2. **特征可视化器** (`feature_visualizer.py`)
   - 专用的语义分割可视化模块
   - 支持7个语义类别的颜色映射
   - 提供置信度分析和统计图表

3. **增强可视化** (`visualizer.py`)
   - 新增 `create_comprehensive_view_with_features()` 方法
   - 自动检测特征并切换到增强模式
   - 支持语义分割叠加和独立显示

4. **智能集成** (`app.py`)
   - 自动识别可用特征
   - 智能选择可视化模式
   - 保存增强版本的可视化结果

## 🎯 语义类别定义

| ID | 类别 | 颜色 | 含义 |
|----|----|------|-----|
| 0 | Background | 黑色 | 背景区域 |
| 1 | Road | 灰色 | 可行驶道路 |
| 2 | Walkways | 棕色 | 人行道 |
| 3 | Centerline | 黄色 | 车道中心线 |
| 4 | Static Objects | 红色 | 静态障碍物 |
| 5 | Vehicles | 蓝色 | 其他车辆 |
| 6 | Pedestrians | 绿色 | 行人 |

## 🚀 快速使用

### 方法1: 使用现有应用
```python
from trajectory_app.app import TrajectoryPredictionApp

# 创建配置
config = {
    "model": {
        "type": "diffusiondrive",
        "checkpoint_path": None,  # 使用默认检查点
        "lr": 6e-4
    },
    "data": {
        "navsim_log_path": "/path/to/navsim_logs/test",
        "sensor_blobs_path": "/path/to/sensor_blobs/test", 
        "cache_path": "/path/to/metric_cache"
    },
    "visualization": {
        "time_windows": [1.0, 3.0, 6.0],
        "save_formats": ["png"]
    }
}

# 初始化应用（自动加载模型和初始化组件）
app = TrajectoryPredictionApp(config)

# 预测并可视化（自动检测特征）
result = app.predict_single_scene(
    scene_token="your_scene_token",
    time_window=(0, 3.0),
    save_visualization=True
)

# 检查是否有特征
if result["visualization"]["has_features"]:
    print(f"✅ 已提取特征: {result['visualization']['feature_types']}")
```

### 方法2: 直接使用特征可视化器
```python
from trajectory_app.feature_visualizer import FeatureVisualizer
import matplotlib.pyplot as plt

# 创建可视化器
feature_viz = FeatureVisualizer()

# 假设你有提取的特征
semantic_data = result["extracted_features"]["bev_semantic_map"]

# 创建语义分割可视化
fig, axes = feature_viz.visualize_bev_semantic_map(
    semantic_data["predictions"],
    confidence_map=semantic_data["confidence"],
    show_legend=True
)
plt.show()
```

## 🧪 测试新功能

我们创建了专用测试脚本来验证所有功能：

```bash
# 从项目根目录运行（重要！）

# 首先验证配置和导入是否正确
python verify_config_fix.py

# 然后运行完整的特征测试
python test_bev_semantic_features.py
```

这些脚本会：
- ✅ 验证配置和导入正确性
- ✅ 测试特征提取功能
- ✅ 创建语义分割可视化
- ✅ 生成综合特征视图
- ✅ 保存测试结果到 `./test_output`

**注意**: 确保你的环境变量 `OPENSCENE_DATA_ROOT` 和 `NAVSIM_EXP_ROOT` 正确设置，或者脚本会使用默认路径。

## 📊 可视化结果

### 标准模式 vs 特征增强模式

**标准模式**：
- BEV轨迹视图
- 前视相机图像
- 轨迹对比图
- 统计面板

**特征增强模式** (NEW!)：
- BEV轨迹 + 语义分割叠加
- 独立的语义分割图
- 预测置信度热力图
- 特征统计分布图

### 文件命名规则

- 标准可视化: `scene_xxx_prediction.png`
- 特征增强版: `scene_xxx_prediction_with_features.png`

## 🔍 技术实现细节

### 特征提取流程
1. 模型前向推理 → 获取 `predictions` 字典
2. 检测 `bev_semantic_map` 输出
3. 应用 `argmax` 获取类别预测
4. 计算 `softmax` 获取置信度
5. 封装为结构化数据

### 坐标系处理
- ✅ 正确处理NavSim的BEV坐标约定
- ✅ Y轴 → matplotlib X轴映射
- ✅ X轴 → matplotlib Y轴映射
- ✅ 与现有轨迹可视化对齐

### 性能优化
- 🚀 非侵入式Hook机制
- 🚀 智能特征缓存
- 🚀 条件性可视化生成
- 🚀 内存友好的处理

## 📈 下一步计划

根据 `FEATURE_VISUALIZATION_TODOLIST.md`:

**即将实现的功能**:
- **Task 1.2**: BEV注意力权重热力图
- **Task 2.1**: 多尺度BEV特征图
- **Task 3.1**: 轨迹Diffusion去噪过程

## 🛠️ 故障排除

### 常见问题

1. **没有提取到特征**
   ```
   ⚠️ 未提取到BEV语义分割特征
   ```
   - 检查模型是否支持BEV语义分割头
   - 确认模型正确加载
   - 查看日志中的特征提取信息

2. **可视化错误**
   - 确保所有依赖正确安装: `matplotlib`, `numpy`, `cv2`
   - 检查输出目录权限
   - 验证特征数据格式

3. **坐标系问题**
   - 我们已经修复了BEV坐标系映射问题
   - 如果仍有问题，参考 `COORDINATE_FIX.md`

## 🎉 成功验证

完成Task 1.1后，你应该能看到：

✅ **可视化结果包含**:
- 彩色的BEV语义分割图
- 轨迹叠加在语义图上
- 置信度热力图
- 详细的类别统计

✅ **日志输出显示**:
```
✅ Extracted BEV semantic map: (128, 128), classes: [0 1 2 3 4 5 6]
✅ Created feature-enhanced visualization with 1 feature types
```

✅ **保存的文件**:
- 增强版轨迹可视化
- 独立的语义分割图
- 综合特征分析图

---

🎊 **恭喜！你已经成功实现了DiffusionDrive模型的第一个特征可视化功能！**

这为后续更高级的特征分析（注意力权重、多尺度特征、Diffusion过程等）奠定了坚实的基础。