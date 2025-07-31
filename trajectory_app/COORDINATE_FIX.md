# 坐标系修复与轨迹投影功能

## 🐛 BEV坐标系问题修复

### 问题描述
轨迹投影到BEV视图时坐标系不匹配，出现错误的对称投影。

### 根本原因
```
用户观察：把trajectory comparison按照BEV的x轴，做了对称投影
```

**原因分析**：
1. **轨迹对比图** 使用标准坐标映射：`(轨迹X, 轨迹Y) → (matplotlib X, matplotlib Y)`
2. **NavSim BEV** 使用特殊坐标映射：`(轨迹Y, 轨迹X) → (matplotlib X, matplotlib Y)`
3. **我们的BEV** 错误地使用了轨迹对比图的映射，然后应用BEV的轴变换

### NavSim BEV坐标系标准

#### 坐标系定义 (`navsim/visualization/plots.py`)
```python
# NOTE: x forward, y sideways
# NOTE: left is y positive, right is y negative
```

#### BEV轨迹绘制 (`navsim/visualization/bev.py`)
```python
def add_trajectory_to_bev_ax(ax, trajectory, config):
    poses = np.concatenate([np.array([[0, 0]]), trajectory.poses[:, :2]])
    ax.plot(
        poses[:, 1],  # 轨迹Y坐标 → matplotlib X轴 (左右方向)
        poses[:, 0],  # 轨迹X坐标 → matplotlib Y轴 (前后方向)
    )
```

#### BEV坐标配置
```python
def configure_bev_ax(ax):
    # X轴显示左右，Y轴显示前后
    ax.set_xlim(-margin_y / 2, margin_y / 2)  # 使用margin_y
    ax.set_ylim(-margin_x / 2, margin_x / 2)  # 使用margin_x
    ax.invert_xaxis()  # 左侧为正
```

### 修复方案

#### 🔥 修复前（错误）
```python
ax.plot(
    filtered_poses[i:i+2, 0],  # 轨迹X → matplotlib X ❌
    filtered_poses[i:i+2, 1],  # 轨迹Y → matplotlib Y ❌
)
```

#### ✅ 修复后（正确）
```python
# 🔥 坐标系修复：NavSim BEV uses (Y, X) mapping
ax.plot(
    filtered_poses[i:i+2, 1],  # 轨迹Y → matplotlib X ✅
    filtered_poses[i:i+2, 0],  # 轨迹X → matplotlib Y ✅
)
```

### 坐标系对比

| 视图 | matplotlib X轴 | matplotlib Y轴 | 视觉效果 |
|------|---------------|---------------|----------|
| **轨迹对比图** | 轨迹 X | 轨迹 Y | X→右, Y→上 |
| **NavSim BEV** | 轨迹 Y | 轨迹 X | X→上, Y→左 |
| **修复后BEV** | 轨迹 Y | 轨迹 X | X→上, Y→左 ✅ |

## 🎯 轨迹投影到前视图功能

### 新功能概述
在前置摄像头图像上实时投影轨迹，包括：
- 模型预测轨迹
- 真实轨迹（Ground Truth）
- PDM-Closed参考轨迹

### 技术实现

#### 1. 坐标变换流程
```
轨迹2D坐标 → 3D世界坐标 → 摄像头坐标系 → 2D图像坐标
```

#### 2. 核心函数
```python
def _add_trajectory_projections_to_image(self, image, camera, trajectories, time_window):
    """将轨迹投影到摄像头图像"""
    
    # 1. 构建3D轨迹点（地面高度=0）
    trajectory_3d = np.zeros((len(poses), 3))
    trajectory_3d[:, :2] = filtered_poses[:, :2]  # X, Y
    trajectory_3d[:, 2] = 0.0  # 地面高度
    
    # 2. 变换到摄像头坐标系
    trajectory_3d_camera = self._transform_trajectory_to_camera_frame(
        trajectory_3d, camera
    )
    
    # 3. 投影到2D图像
    projected_points, in_fov_mask = _transform_points_to_image(
        trajectory_3d_camera, camera.intrinsics, image_shape
    )
    
    # 4. 绘制轨迹线和标记点
    cv2.line(image, pt1, pt2, color_bgr, thickness)
    cv2.circle(image, center, radius, color_bgr, -1)
```

#### 3. 坐标系变换
```python
def _transform_trajectory_to_camera_frame(self, trajectory_3d, camera):
    """从车辆坐标系变换到摄像头坐标系"""
    
    # 获取变换矩阵（利用NavSim的摄像头标定）
    lidar2cam_r = np.linalg.inv(camera.sensor2lidar_rotation)
    lidar2cam_t = camera.sensor2lidar_translation @ lidar2cam_r.T
    
    # 4x4变换矩阵
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    
    # 应用变换
    trajectory_cam = (lidar2cam_rt.T @ trajectory_4d.T).T
    return trajectory_cam[:, :3]
```

### 视觉效果

#### 轨迹样式
- **预测轨迹**：红色实线，圆形标记
- **真实轨迹**：绿色实线，方形标记  
- **PDM-Closed**：蓝色虚线，三角标记

#### 动态效果
- **时间渐变**：未来的轨迹点透明度降低
- **线条粗细**：根据透明度调整线条粗细
- **标记点**：关键点用白色边框突出显示

#### 图例显示
- 右上角显示轨迹类型图例
- 半透明黑色背景，白色文字
- 实时更新可见轨迹类型

## 🚀 使用说明

### 自动应用
修复后的功能会自动应用到：
- `TrajectoryPredictionApp.predict_single_scene()`
- `TrajectoryVisualizer.create_comprehensive_view()`

### 验证修复
1. **BEV坐标系**：轨迹应该正确显示，与轨迹对比图逆时针90°一致
2. **前视图投影**：轨迹应该投影在道路上，符合透视效果

### 调试信息
```python
# 在日志中查看投影信息
logger.debug(f"Projected {len(valid_points)} trajectory points to camera")
logger.warning(f"Failed to project trajectory {traj_name}: {error}")
```

## 🎯 技术亮点

### 1. 完全兼容NavSim
- 使用NavSim原生的投影函数 `_transform_points_to_image`
- 遵循NavSim的坐标系标准
- 利用现有的摄像头标定参数

### 2. 高性能实现
- 批量坐标变换
- 高效的OpenCV绘制
- 智能的视野剪裁

### 3. 视觉优化
- 时间相关的透明度渐变
- 多层次的视觉标记
- 自适应的图例显示

### 4. 错误处理
- 摄像头数据缺失保护
- 投影失败的优雅处理  
- 详细的调试日志

## 📊 修复对比

| 方面 | 修复前 | 修复后 |
|------|--------|--------|
| **BEV坐标** | X轴对称错误 | 正确的90°旋转 |
| **前视图** | 无轨迹投影 | 实时轨迹投影 |
| **视觉效果** | 单一视角 | 多视角融合 |
| **用户体验** | 混淆 | 直观理解 |

这次修复不仅解决了坐标系问题，还大大增强了轨迹可视化的功能和用户体验！🎉 