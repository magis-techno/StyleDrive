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
# 使用脚本运行（推荐）
./scripts/evaluation/run_pdm_score_with_visualization.sh /path/to/checkpoint.ckpt diffusiondrive_style_agent

# 或者直接使用Python命令
python navsim/planning/script/run_pdm_score_with_visualization.py \
    --config-path="config/pdm_scoring" \
    --config-name="default_run_pdm_score_with_visualization" \
    train_test_split=styletest \
    agent=diffusiondrive_style_agent \
    agent.checkpoint_path=/path/to/checkpoint.ckpt \
    enable_visualization=true \
    max_visualizations=20
```

### 2. 配置选项

关键配置参数：

```yaml
# 启用可视化
enable_visualization: true

# 可视化输出目录
visualization_output_dir: "pdm_evaluation_visualizations"

# 最大可视化数量（避免生成过多图片）
max_visualizations: 50

# 使用单线程执行（确保可视化兼容性）
worker: sequential
```

### 3. 输出结果

运行后会生成以下文件：

```
output_dir/
├── pdm_evaluation_visualizations/
│   ├── abcd1234efgh_evaluation.png           # BEV可视化图片
│   ├── abcd1234efgh_evaluation_data.json     # 详细评测数据
│   ├── abcd1234efgh_trajectories.pkl         # 轨迹数据
│   └── visualization_summary.csv             # 可视化场景汇总
└── eval_with_visualization_pdm_score.csv     # 完整评测结果
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

### 核心函数

- `add_pdm_results_to_bev_ax()`: 添加PDM结果文本框
- `add_trajectory_comparison_to_bev_ax()`: 添加多轨迹对比
- `create_evaluation_visualization()`: 创建完整评测可视化
- `save_evaluation_results()`: 保存可视化结果和数据

### 复用的现有组件

- `add_configured_bev_on_ax()`: BEV背景渲染
- `add_annotations_to_bev_ax()`: 车辆标注显示
- `add_trajectory_to_bev_ax()`: 轨迹绘制
- `configure_bev_ax()`: BEV坐标系配置

充分利用了现有的navsim可视化架构，保持了代码的一致性和可维护性。