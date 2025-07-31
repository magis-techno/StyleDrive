#!/usr/bin/env python3
"""
快速测试脚本 - 验证轨迹预测应用的所有修复

运行此脚本来快速检查是否所有已知问题都已修复。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_data_structure_fixes():
    """测试数据结构修复"""
    print("🔍 测试数据结构修复...")
    
    try:
        from navsim.common.dataclasses import EgoStatus, Frame, SceneMetadata
        
        # 检查EgoStatus没有timestamp
        ego_fields = list(EgoStatus.__dataclass_fields__.keys())
        has_timestamp = 'timestamp' in ego_fields
        
        # 检查Frame有timestamp
        frame_fields = list(Frame.__dataclass_fields__.keys())
        frame_has_timestamp = 'timestamp' in frame_fields
        
        # 检查SceneMetadata没有scenario_type
        metadata_fields = list(SceneMetadata.__dataclass_fields__.keys())
        has_scenario_type = 'scenario_type' in metadata_fields
        
        print(f"  ✅ EgoStatus 没有 timestamp: {not has_timestamp}")
        print(f"  ✅ Frame 有 timestamp: {frame_has_timestamp}")
        print(f"  ✅ SceneMetadata 没有 scenario_type: {not has_scenario_type}")
        
        return not has_timestamp and frame_has_timestamp and not has_scenario_type
        
    except Exception as e:
        print(f"  ❌ 数据结构测试失败: {e}")
        return False

def test_code_fixes():
    """测试代码修复"""
    print("\n🔧 测试代码修复...")
    
    try:
        # 测试数据管理器修复
        from trajectory_app.data_manager import TrajectoryDataManager
        import inspect
        
        dm_source = inspect.getsource(TrajectoryDataManager.load_scene_data)
        
        dm_fixes = {
            'scenario_type 修复': '"scenario_type": "unknown"' in dm_source,
            'timestamp 修复': 'current_frame.timestamp' in dm_source,
            '移除错误的 ego_status.timestamp': 'ego_status.timestamp' not in dm_source,
            '修复注释存在': 'Fixed: timestamp is in Frame' in dm_source
        }
        
        # 测试推理引擎修复
        from trajectory_app.inference_engine import TrajectoryInferenceEngine
        
        ie_source = inspect.getsource(TrajectoryInferenceEngine.predict_trajectory)
        
        ie_fixes = {
            '设备转移修复': 'features = {k: v.to(self.device) for k, v in features.items()}' in ie_source,
            'SOLUTION 1 注释': 'SOLUTION 1: Handle device mismatch' in ie_source,
            '设备日志': 'Model device:' in ie_source,
            '错误处理': 'except Exception as e:' in ie_source
        }
        
        # 测试可视化器修复
        from trajectory_app.visualizer import TrajectoryVisualizer
        
        viz_source = inspect.getsource(TrajectoryVisualizer._render_bev_trajectories)
        
        viz_fixes = {
            'BEV坐标系修复': 'filtered_poses[i:i+2, 1]' in viz_source and 'filtered_poses[i:i+2, 0]' in viz_source,
            '坐标系修复注释': '坐标系修复：NavSim BEV uses (Y, X) mapping' in viz_source,
            '轨迹投影功能': hasattr(TrajectoryVisualizer, '_add_trajectory_projections_to_image'),
            '摄像头坐标变换': hasattr(TrajectoryVisualizer, '_transform_trajectory_to_camera_frame')
        }
        
        all_fixes = {**dm_fixes, **ie_fixes, **viz_fixes}
        
        print("  数据管理器修复:")
        for fix_name, passed in dm_fixes.items():
            status = "✅" if passed else "❌"
            print(f"    {status} {fix_name}")
        
        print("  推理引擎修复:")
        for fix_name, passed in ie_fixes.items():
            status = "✅" if passed else "❌"
            print(f"    {status} {fix_name}")
        
        print("  可视化器修复:")
        for fix_name, passed in viz_fixes.items():
            status = "✅" if passed else "❌"
            print(f"    {status} {fix_name}")
        
        return all(all_fixes.values())
        
    except Exception as e:
        print(f"  ❌ 代码修复测试失败: {e}")
        return False

def test_import():
    """测试导入"""
    print("\n📦 测试导入...")
    
    try:
        from trajectory_app import TrajectoryPredictionApp
        print("  ✅ 主应用导入成功")
        
        from trajectory_app.data_manager import TrajectoryDataManager
        print("  ✅ 数据管理器导入成功")
        
        from trajectory_app.visualizer import TrajectoryVisualizer
        print("  ✅ 可视化器导入成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False

def test_device_setup():
    """测试设备设置"""
    print("\n🖥️ 测试设备设置...")
    
    try:
        import torch
        
        print(f"  📝 PyTorch 版本: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"  🔧 CUDA 可用: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            print(f"  📱 GPU 数量: {device_count}")
            print(f"  📱 当前 GPU: {current_device} ({device_name})")
            print(f"  📱 CUDA 版本: {torch.version.cuda}")
            
            # 测试设备创建
            device = torch.device("cuda" if cuda_available else "cpu")
            print(f"  ✅ 推荐设备: {device}")
            
            return True
        else:
            print("  ℹ️ 将使用 CPU 模式")
            return True
            
    except Exception as e:
        print(f"  ❌ 设备测试失败: {e}")
        return False

def test_environment():
    """测试环境设置"""
    print("\n🌍 测试环境设置...")
    
    openscene_root = os.environ.get('OPENSCENE_DATA_ROOT')
    navsim_root = os.environ.get('NAVSIM_EXP_ROOT')
    
    print(f"  📝 OPENSCENE_DATA_ROOT: {openscene_root or 'NOT SET'}")
    print(f"  📝 NAVSIM_EXP_ROOT: {navsim_root or 'NOT SET'}")
    
    if openscene_root and navsim_root:
        # 检查路径存在
        data_path = Path(openscene_root) / "navsim_logs" / "test"
        sensor_path = Path(openscene_root) / "sensor_blobs" / "test"
        
        data_exists = data_path.exists()
        sensor_exists = sensor_path.exists()
        
        print(f"  📁 数据路径存在: {data_exists} ({data_path})")
        print(f"  📁 传感器路径存在: {sensor_exists} ({sensor_path})")
        
        return data_exists and sensor_exists
    else:
        print("  ⚠️ 环境变量未完全设置")
        return False

def main():
    """主函数"""
    print("🚀 轨迹预测应用快速测试")
    print("=" * 60)
    
    tests = [
        ("数据结构修复", test_data_structure_fixes),
        ("代码修复", test_code_fixes), 
        ("导入测试", test_import),
        ("设备设置", test_device_setup),
        ("环境设置", test_environment)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
            print(f"  🎉 {test_name} 通过")
        else:
            print(f"  ❌ {test_name} 失败")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 项通过")
    
    if passed == total:
        print("🎉 所有测试通过! 可以运行 tutorial notebook 了!")
        
        print("\n🚀 下一步:")
        print("  1. 重启 Jupyter Kernel (如果还没有)")
        print("  2. 运行 tutorial notebook")
        print("  3. 从第一个 cell 开始逐个运行")
        
    elif passed >= total - 1:
        print("⚠️ 大部分测试通过，可以尝试运行 tutorial")
        print("  如果仍有问题，请检查失败的测试项")
        
    else:
        print("❌ 多项测试失败，建议先修复问题")
        
        print("\n🔧 建议的修复步骤:")
        if passed < 3:
            print("  1. 重启 Jupyter Kernel")
            print("  2. 检查环境变量设置")
            print("  3. 确保在正确的项目目录")
        print("  4. 运行数据结构探索: python trajectory_app/explore_data_structure.py")
        print("  5. 查看详细错误信息")

if __name__ == "__main__":
    main() 