#!/usr/bin/env python3
"""
测试BEV可视化功能的简单脚本

这个脚本可以帮助验证新的BEV可视化功能是否正确实现。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def test_imports():
    """测试所有必要的导入"""
    print("🧪 测试导入...")
    
    try:
        from style_trajectory_app import plot_style_trajectories_bev
        print("✅ plot_style_trajectories_bev 导入成功")
    except ImportError as e:
        print(f"❌ plot_style_trajectories_bev 导入失败: {e}")
        return False
    
    try:
        from style_trajectory_app import StyleTrajectoryApp
        print("✅ StyleTrajectoryApp 导入成功")
    except ImportError as e:
        print(f"❌ StyleTrajectoryApp 导入失败: {e}")
        return False
    
    try:
        from style_trajectory_app.cli import parse_arguments
        print("✅ CLI参数解析 导入成功")
    except ImportError as e:
        print(f"❌ CLI参数解析 导入失败: {e}")
        return False
    
    return True

def test_view_type_parameter():
    """测试view_type参数"""
    print("\n🧪 测试CLI参数...")
    
    try:
        from style_trajectory_app.cli import parse_arguments
        import argparse
        
        # 模拟命令行参数
        test_args = [
            '--checkpoint', 'dummy.ckpt',
            '--split', 'navtest', 
            '--view-type', 'bev'
        ]
        
        # 备份原始sys.argv
        original_argv = sys.argv
        sys.argv = ['test'] + test_args
        
        try:
            args = parse_arguments()
            print(f"✅ 参数解析成功: view_type = {args.view_type}")
            if args.view_type == 'bev':
                print("✅ BEV视图类型设置正确")
            else:
                print(f"❌ BEV视图类型设置错误: {args.view_type}")
                return False
        finally:
            # 恢复原始sys.argv
            sys.argv = original_argv
            
    except Exception as e:
        print(f"❌ CLI参数测试失败: {e}")
        return False
    
    return True

def test_app_initialization():
    """测试应用初始化（不需要实际模型）"""
    print("\n🧪 测试应用接口...")
    
    try:
        from style_trajectory_app import StyleTrajectoryApp
        
        # 检查run_style_demo方法签名
        import inspect
        signature = inspect.signature(StyleTrajectoryApp.run_style_demo)
        params = list(signature.parameters.keys())
        
        if 'view_type' in params:
            print("✅ run_style_demo 方法包含 view_type 参数")
        else:
            print(f"❌ run_style_demo 方法缺少 view_type 参数: {params}")
            return False
            
        # 检查默认值
        view_type_param = signature.parameters.get('view_type')
        if view_type_param and view_type_param.default == 'simple':
            print("✅ view_type 默认值设置正确 (simple)")
        else:
            print(f"❌ view_type 默认值错误: {view_type_param.default if view_type_param else 'None'}")
            return False
            
    except Exception as e:
        print(f"❌ 应用接口测试失败: {e}")
        return False
    
    return True

def test_visualization_functions():
    """测试可视化函数"""
    print("\n🧪 测试可视化函数...")
    
    try:
        from style_trajectory_app.style_visualization import (
            plot_style_trajectories_bev,
            plot_style_trajectories_simple_fallback,
            NAVSIM_VIZ_AVAILABLE
        )
        
        print(f"✅ BEV可视化函数导入成功")
        print(f"ℹ️  NavSim可视化可用性: {NAVSIM_VIZ_AVAILABLE}")
        
        # 检查函数签名
        import inspect
        bev_signature = inspect.signature(plot_style_trajectories_bev)
        bev_params = list(bev_signature.parameters.keys())
        
        expected_params = ['trajectories', 'scene', 'frame_idx', 'title', 'figsize']
        for param in expected_params:
            if param in bev_params:
                print(f"✅ plot_style_trajectories_bev 包含参数: {param}")
            else:
                print(f"❌ plot_style_trajectories_bev 缺少参数: {param}")
                return False
                
    except ImportError as e:
        print(f"❌ 可视化函数导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 可视化函数测试失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始测试BEV可视化功能")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_imports),
        ("CLI参数测试", test_view_type_parameter), 
        ("应用接口测试", test_app_initialization),
        ("可视化函数测试", test_visualization_functions)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 发生异常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！BEV可视化功能实现正确。")
        print("\n📝 使用示例:")
        print("  python -m style_trajectory_app.cli -c model.ckpt --split navtest --view-type bev")
    else:
        print("⚠️  部分测试失败，请检查实现。")
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)