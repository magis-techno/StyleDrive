# PDM 评测结果可视化指南

## 概述

本功能在现有的 `navsim.visualization` 基础上，增加了PDM评测结果的可视化显示，可以直观地查看评测对象的轨迹预测质量和各项指标得分。

## 功能特性

### ✨ 主要功能
- **BEV可视化集成**：在鸟瞰图中显示地图、标注和轨迹
- **多轨迹对比**：同时显示预测轨迹、真实轨迹和PDM参考轨迹
- **评测结果显示**：以文本框形式显示详细的PDM各项指标
- **Style识别**：支持aggressive、normal、conservative等驾驶风格的可视化
- **批量处理**：支持对多个场景进行评测和可视化

### 📊 显示内容
- **总体评分**：PDM综合得分
- **安全指标**：碰撞检测、可行驶区域合规性
- **效率指标**：行程进度、行驶方向合规性  
- **舒适指标**：舒适度、碰撞时间预警

## 文件结构

新增的文件如下：

```
navsim/
├── visualization/
│   ├── evaluation_viz.py              # 核心可视化模块
│   └── config.py                      # 扩展配置（新增PDM_RESULTS_CONFIG等）
└── planning/script/
    ├── run_pdm_score_with_visualization.py    # 集成评测脚本
    └── config/pdm_scoring/
        └── default_run_pdm_score_with_visualization.yaml

scripts/evaluation/
└── run_pdm_score_with_visualization.sh       # 便捷运行脚本

test_evaluation_visualization.py              # 测试脚本
docs/evaluation_visualization_guide.md        # 本文档
```

## 使用方法

### 1. 基本评测与可视化

运行带有可视化的PDM评测：

```bash
# 使用便捷脚本（推荐）
./scripts/evaluation/run_pdm_score_with_viz.sh /path/to/checkpoint.ckpt diffusiondrive_style_agent 20

# 或者直接使用现有的PDM评测脚本
python navsim/planning/script/run_pdm_score.py \
    train_test_split=styletest \
    agent=diffusiondrive_style_agent \
    agent.checkpoint_path=/path/to/checkpoint.ckpt \
    experiment_name=eval_with_viz \
    +enable_visualization=true \
    +max_visualizations=20
```

### 2. 配置选项

关键配置参数（通过命令行传递）：

```bash
# 启用可视化（注意+前缀，表示添加新配置）
+enable_visualization=true

# 最大可视化数量（避免生成过多图片，推荐20-50）
+max_visualizations=20

# 实验名称（影响输出目录）
experiment_name=eval_with_viz
```

**注意**：可视化功能自动与任何worker配置兼容，无需特殊设置。

### 3. 输出结果

运行后会生成以下文件：

```
output_dir/
├── visualizations/
│   ├── abcd1234efgh_evaluation.png           # BEV可视化图片
│   ├── abcd1234efgh_evaluation_data.json     # 详细评测数据
│   └── abcd1234efgh_trajectories.pkl         # 轨迹数据
├── traj_and_metric/                          # 原有轨迹保存
│   └── abcd1234efgh.pkl
└── {timestamp}.csv                           # 完整评测结果（含可视化路径）
```

### 4. 程序化使用

也可以在Python代码中直接使用可视化功能：

```python
from navsim.visualization.evaluation_viz import create_evaluation_visualization, save_evaluation_results
from navsim.common.dataclasses import PDMResults

# 准备数据
trajectories = {
    "predicted": predicted_trajectory,
    "ground_truth": gt_trajectory,
    "pdm_reference": pdm_trajectory
}

pdm_results = {
    "predicted": pdm_result
}

# 创建可视化
fig = create_evaluation_visualization(
    frame=current_frame,
    trajectories=trajectories,
    pdm_results=pdm_results,
    map_api=map_api,
    scene_token=scene_token,
    style="aggressive"
)

# 保存结果
evaluation_results = {
    "figure": fig,
    "trajectories": trajectories,
    "pdm_results": pdm_results,
    "style": "aggressive"
}

save_evaluation_results(evaluation_results, output_dir, scene_token)
```

## 坐标系说明

本可视化遵循navsim的标准坐标系：
- **向上是X的正方向**（车辆前进方向）
- **向左是Y的正方向**（车辆左侧方向）
- 在BEV图中，ego车辆位于图像中心，朝向上方

## 可视化元素

### 轨迹显示
- **预测轨迹**：红色实线，方形标记
- **真实轨迹**：绿色实线，圆形标记  
- **PDM参考轨迹**：蓝色虚线，三角标记

### 评测结果文本框
位置：右上角（可配置）
内容：
```
PDM Score: 0.847
Style: Aggressive

Safety:
  ✓ No Collision: 0.950
  ✓ Drivable Area: 0.920

Efficiency:
  Progress: 0.880
  Direction: 0.900

Comfort:
  Comfort: 0.750
  TTC: 0.820
```

## 测试验证

运行测试脚本验证功能：

```bash
python test_evaluation_visualization.py
```

测试将生成多个示例图片，验证：
- 基本PDM结果显示
- 多轨迹对比
- 完整评测可视化
- 坐标系正确性
- 不同驾驶风格对比

## 性能考虑

- **内存管理**：自动关闭matplotlib图形释放内存
- **批量限制**：通过`max_visualizations`限制生成数量
- **单线程执行**：确保可视化兼容性，避免并发问题

## 故障排除

### 常见问题

1. **导入错误**：确保安装了matplotlib和相关依赖
2. **内存不足**：减少`max_visualizations`数量
3. **可视化失败**：检查是否有map_api，可视化会自动降级到仅显示标注

### 日志信息

关键日志：
```
INFO: Generated visualization for scene abcd1234
WARNING: Map API not available for scene abcd1234
WARNING: Failed to generate visualization for abcd1234: [error details]
```

## 扩展与定制

### 自定义显示样式

修改 `navsim/visualization/config.py` 中的配置：

```python
PDM_RESULTS_CONFIG = {
    "text_box": {
        "position": "top-left",  # 改变文本框位置
        "bbox": {
            "facecolor": "lightblue",  # 改变背景色
            "alpha": 0.8,
        },
        "font_size": 12,  # 改变字体大小
    }
}
```

### 添加新的轨迹类型

在 `TRAJECTORY_CONFIG` 中添加新配置：

```python
TRAJECTORY_CONFIG = {
    "my_custom_trajectory": {
        "line_color": "#FF5733",
        "line_width": 3.0,
        "marker": "d",
        # ... 其他配置
    }
}
```

## 技术细节

### 集成架构

本可视化功能采用**最小侵入式设计**，直接集成到现有的 `run_pdm_score.py` 中：

- ✅ **零配置冲突**：完全兼容现有配置系统
- ✅ **可选加载**：可视化模块缺失时自动降级
- ✅ **内存管理**：自动释放matplotlib资源
- ✅ **错误隔离**：可视化失败不影响评测

### 核心函数

- `create_evaluation_visualization()`: 创建完整评测可视化
- `save_evaluation_results()`: 保存可视化结果和数据
- `add_pdm_results_to_bev_ax()`: 添加PDM结果文本框
- `add_trajectory_comparison_to_bev_ax()`: 添加多轨迹对比

### 复用的现有组件

- `add_configured_bev_on_ax()`: BEV背景渲染
- `add_annotations_to_bev_ax()`: 车辆标注显示
- `add_trajectory_to_bev_ax()`: 轨迹绘制
- `configure_bev_ax()`: BEV坐标系配置

**优势**：充分利用现有架构，保持代码一致性，便于维护。

## 🎨 多风格对比功能

### 可视化内容说明

当前可视化显示3条轨迹：

1. **🔴 Predicted (红色)**：模型对当前场景的预测轨迹
2. **🟢 Ground Truth (绿色)**：人类驾驶员的实际行驶轨迹  
3. **🔵 PDM Reference (蓝色)**：PDM系统生成的参考轨迹（用于评分基准）

**Style含义**：显示的是数据集中该场景的真实驾驶风格标注。

### 多风格对比

想要看同一个场景在不同风格下的预测结果？使用多风格对比功能：

```bash
# 对单个场景进行多风格对比
./scripts/evaluation/run_multi_style_comparison.sh /path/to/checkpoint.ckpt scene_token_12345

# 这会生成一个包含多个子图的对比可视化：
# - 总览图：所有风格轨迹在一起对比
# - 分图：每种风格的详细评测结果
```

**输出效果**：
- 左侧：总览对比（所有风格轨迹叠加）
- 右侧：各风格的独立详细分析（包含PDM评分）

### 程序化多风格分析

```python
from navsim.visualization.multi_style_viz import run_multi_style_evaluation

# 对同一场景运行多种风格
results = run_multi_style_evaluation(
    agent_input=agent_input,
    scene=scene, 
    agent=agent,
    styles=["aggressive", "normal", "conservative"]
)

# 结果包含每种风格的轨迹和PDM评分
for style, data in results["style_results"].items():
    print(f"{style}: Score = {data['pdm_result'].score:.3f}")
```