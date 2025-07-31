# 轨迹预测应用 (Trajectory Prediction Application)

一个完整的轨迹预测推理和可视化应用，支持 DiffusionDrive 和 Transfuser 模型的模型推理、轨迹对比和综合可视化。

## 🚀 功能特性

- **🤖 多模型支持**: DiffusionDrive, Transfuser
- **📊 丰富可视化**: BEV视图（已修复坐标系）、前视图轨迹投影、轨迹对比、统计面板
- **🎯 轨迹投影**: 实时将轨迹投影到前置摄像头图像上（新功能）
- **📈 评估指标**: ADE, FDE, RMSE 等标准轨迹评估指标
- **⚡ 批量处理**: 支持多场景批量推理和评估
- **🎯 时间窗口**: 灵活的时间范围可视化 (1.5s, 3s, 6s等)
- **🔧 设备优化**: 智能GPU/CPU设备管理，自动解决设备不匹配问题
- **🔧 易于配置**: YAML配置文件和代码配置支持
- **💾 结果保存**: 自动保存可视化图片和统计报告

## 📁 项目结构

```
trajectory_app/
├── __init__.py                    # 包初始化
├── app.py                         # 主应用类
├── inference_engine.py            # 模型推理引擎
├── data_manager.py               # 数据管理器
├── visualizer.py                 # 可视化器
├── config/
│   └── default_config.yaml      # 默认配置文件
├── tutorial/
│   └── trajectory_prediction_tutorial.ipynb  # 教程Notebook
└── README.md                     # 本文档
```

## 🛠️ 安装和设置

### 1. 环境要求

- Python 3.8+
- PyTorch
- NavSim 开发工具包
- matplotlib, numpy, yaml, opencv-python

### 2. 环境变量设置

在运行应用之前，请设置以下环境变量：

```bash
export OPENSCENE_DATA_ROOT="/path/to/your/openscene/data"
export NAVSIM_EXP_ROOT="/path/to/your/navsim/experiments"
```

### 3. 数据准备

确保以下路径存在：
- `$OPENSCENE_DATA_ROOT/navsim_logs/test/` - NavSim 日志数据
- `$OPENSCENE_DATA_ROOT/sensor_blobs/test/` - 传感器数据
- `$NAVSIM_EXP_ROOT/metric_cache/` - 评估缓存数据（可选）

### 4. 模型权重

准备训练好的模型权重文件（.pth格式），或使用随机初始化权重进行测试。

## 🚀 快速开始

### 方法 1: 使用 Jupyter Notebook （推荐）

1. 打开教程 Notebook：
```bash
jupyter notebook trajectory_app/tutorial/trajectory_prediction_tutorial.ipynb
```

2. 按照教程逐步运行，包含完整的示例和说明

### 方法 2: Python 脚本

```python
import os
from trajectory_app import TrajectoryPredictionApp

# 设置环境变量
os.environ["OPENSCENE_DATA_ROOT"] = "/path/to/your/data"
os.environ["NAVSIM_EXP_ROOT"] = "/path/to/your/experiments"

# 配置应用
config = {
    "model": {
        "type": "diffusiondrive",
        "checkpoint_path": "/path/to/your/model.pth",
        "lr": 6e-4
    },
    "data": {
        "navsim_log_path": f"{os.environ['OPENSCENE_DATA_ROOT']}/navsim_logs/test",
        "sensor_blobs_path": f"{os.environ['OPENSCENE_DATA_ROOT']}/sensor_blobs/test",
        "cache_path": f"{os.environ['NAVSIM_EXP_ROOT']}/metric_cache"
    },
    "output": {
        "output_dir": "./output"
    }
}

# 初始化应用
app = TrajectoryPredictionApp(config)

# 获取测试场景
test_scenes = app.get_random_scenes(num_scenes=3)

# 单场景预测
result = app.predict_single_scene(
    scene_token=test_scenes[0],
    time_window=(0, 3.0),
    save_visualization=True
)

print(f"ADE: {result['metrics']['ade']:.2f}m")
print(f"FDE: {result['metrics']['fde']:.2f}m")
```

## 🎯 主要应用场景

### 1. 单场景轨迹预测
对单个驾驶场景进行轨迹预测，生成综合可视化包含：
- BEV 鸟瞰图轨迹对比
- 前视摄像头视图
- 轨迹误差分析
- 详细统计信息

### 2. 时间窗口比较
比较不同时间跨度（1.5s, 3s, 6s）的轨迹预测效果，分析模型在不同预测范围的性能。

### 3. 批量场景评估
批量处理多个场景，生成：
- 汇总统计报告（YAML格式）
- 按场景类型分组的性能指标
- 批量可视化结果

### 4. 模型对比
支持在相同场景下对比不同模型（DiffusionDrive vs Transfuser）的预测效果。

## 📊 输出结果

### 可视化文件
- **综合视图**: 包含BEV、前视图、统计的大图
- **简单BEV图**: 专注于轨迹对比的鸟瞰图
- **时间窗口对比图**: 并排显示不同时间跨度

### 统计报告
- **batch_summary.yaml**: 批量处理的详细统计
- **场景级指标**: ADE, FDE, RMSE 等
- **按场景类型统计**: 不同驾驶场景的性能分析

### 评估指标
- **ADE (Average Displacement Error)**: 平均位移误差
- **FDE (Final Displacement Error)**: 最终位移误差  
- **RMSE (Root Mean Square Error)**: 均方根误差
- **Max Error**: 最大误差

## ⚙️ 配置选项

### 模型配置
```yaml
model:
  type: "diffusiondrive"  # 或 "transfuser"
  checkpoint_path: "/path/to/model.pth"
  lr: 6e-4
```

### 可视化配置
```yaml
visualization:
  time_windows: [1.0, 3.0, 6.0]
  trajectory_styles:
    prediction:
      color: "#DC143C"
      style: "-"
      width: 3
    ground_truth:
      color: "#2E8B57"
      style: "-" 
      width: 3
    pdm_closed:
      color: "#4169E1"
      style: "--"
      width: 2
```

## 🔧 扩展开发

### 添加新模型类型
1. 在 `inference_engine.py` 中添加新的 `_load_your_model` 方法
2. 确保模型实现 `AbstractAgent` 接口
3. 更新配置文件支持新的模型类型

### 自定义可视化
1. 修改 `visualizer.py` 中的 `trajectory_styles`
2. 添加新的渲染方法
3. 扩展 `create_comprehensive_view` 函数

### 新的评估指标
1. 在 `visualizer.py` 的 `_calculate_trajectory_metrics` 中添加新指标
2. 更新可视化显示相应指标

## 🐛 故障排除

### 常见问题

1. **ModuleNotFoundError: No module named 'trajectory_app'**
   - 确保在项目根目录运行
   - 检查 Python 路径设置

2. **数据路径不存在**
   - 验证环境变量 OPENSCENE_DATA_ROOT
   - 确认数据目录结构正确

3. **模型加载失败**
   - 检查模型权重文件路径
   - 确认模型类型配置正确

4. **CUDA/GPU 相关错误**
   - 检查 PyTorch GPU 支持
   - 可设置为 CPU 模式进行测试

### 性能优化

1. **内存使用**
   - 减小批量处理大小
   - 关闭不必要的可视化

2. **处理速度**
   - 使用 GPU 加速
   - 启用数据缓存

## 📚 相关资源

- [NavSim 官方文档](https://github.com/autonomousvision/navsim)
- [DiffusionDrive 论文](https://arxiv.org/abs/2411.15139)
- [Transfuser 模型](https://github.com/autonomousvision/transfuser)

## 📄 许可证

本项目使用与 NavSim 相同的许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个应用！

---

**Happy trajectory prediction! 🚗✨** 