#!/usr/bin/env python3
"""
StyleTrajectoryApp Command Line Interface

Usage:
    python -m style_trajectory_app.cli --checkpoint /path/to/model.ckpt --dataset /path/to/dataset
    
    # 或者直接运行
    python style_trajectory_app/cli.py --checkpoint /path/to/model.ckpt --dataset /path/to/dataset
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

from .app import StyleTrajectoryApp


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="StyleTrajectoryApp - 风格化轨迹预测命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python -m style_trajectory_app.cli --checkpoint model.ckpt --split navtest  
  python -m style_trajectory_app.cli -c model.ckpt -s navmini --output results/
  python -m style_trajectory_app.cli -c model.ckpt -s styletrain --scenes 5
        """
    )
    
    parser.add_argument(
        '--checkpoint', '-c', 
        type=str, 
        required=True,
        help='DiffusionDrive-Style模型检查点路径'
    )
    
    parser.add_argument(
        '--split', '-sp',
        type=str, 
        default='navtest',
        help='数据集split名称 (默认: navtest, 可选: navmini, styletrain等)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./style_trajectory_results',
        help='输出目录 (默认: ./style_trajectory_results)'
    )
    
    parser.add_argument(
        '--scenes', '-s',
        type=int,
        default=1,
        help='要处理的场景数量 (默认: 1)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=6e-4,
        help='学习率 (默认: 6e-4)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )
    
    return parser.parse_args()


def setup_output_directory(output_path: str) -> Path:
    """设置输出目录"""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    (output_dir / 'data').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    
    return output_dir


def save_results(demo_result: Dict[str, Any], output_dir: Path, scene_idx: int):
    """保存演示结果"""
    scene_token = demo_result['scene_token']
    
    # 保存可视化图片
    fig = demo_result['visualization']
    viz_path = output_dir / 'visualizations' / f'scene_{scene_idx:03d}_{scene_token[:8]}.png'
    fig.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 释放内存
    
    # 保存轨迹数据
    trajectories_data = {}
    for style_name, trajectory_tensor in demo_result['trajectories'].items():
        # 转换为numpy数组并保存
        traj_np = trajectory_tensor.detach().cpu().numpy()[0]  # 移除batch维度
        trajectories_data[style_name] = {
            'trajectory': traj_np.tolist(),
            'shape': list(traj_np.shape),
            'start_point': [float(traj_np[0, 0]), float(traj_np[0, 1])],
            'end_point': [float(traj_np[-1, 0]), float(traj_np[-1, 1])],
            'total_distance': float(
                sum([
                    ((traj_np[i+1, 0] - traj_np[i, 0])**2 + (traj_np[i+1, 1] - traj_np[i, 1])**2)**0.5
                    for i in range(len(traj_np)-1)
                ])
            )
        }
    
    # 保存场景元数据和结果
    result_data = {
        'scene_token': scene_token,
        'scene_metadata': demo_result['scene_metadata'],
        'trajectories': trajectories_data,
        'timing': {
            'prediction_time': demo_result['prediction_time'],
            'total_demo_time': demo_result['demo_time']
        },
        'files': {
            'visualization': str(viz_path.name),
            'data_file': f'scene_{scene_idx:03d}_{scene_token[:8]}_data.json'
        }
    }
    
    data_path = output_dir / 'data' / f'scene_{scene_idx:03d}_{scene_token[:8]}_data.json'
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    return viz_path, data_path


def run_style_demo_cli(args):
    """运行命令行风格演示"""
    import numpy as np
    np.random.seed(args.seed)
    
    print("🚀 启动StyleTrajectoryApp命令行版本")
    print("=" * 60)
    
    # 验证输入路径
    if not Path(args.checkpoint).exists():
        print(f"❌ 检查点文件不存在: {args.checkpoint}")
        return 1
        
    # 检查环境变量
    openscene_root = os.environ.get('OPENSCENE_DATA_ROOT')
    if not openscene_root:
        print(f"❌ 环境变量 OPENSCENE_DATA_ROOT 未设置")
        print(f"请设置环境变量指向数据集根目录")
        return 1
    
    if not Path(openscene_root).exists():
        print(f"❌ 数据集根目录不存在: {openscene_root}")
        return 1
    
    # 设置输出目录
    output_dir = setup_output_directory(args.output)
    print(f"📁 输出目录: {output_dir.absolute()}")
    
    # 初始化应用
    print(f"\n🔧 初始化应用...")
    print(f"  - 检查点: {args.checkpoint}")
    print(f"  - 数据集split: {args.split}")
    print(f"  - 数据集根目录: {openscene_root}")
    print(f"  - 学习率: {args.lr}")
    
    try:
        app = StyleTrajectoryApp(
            checkpoint_path=args.checkpoint,
            split=args.split,
            lr=args.lr
        )
        print("✅ 应用初始化成功!")
        
        if args.verbose:
            print(f"\n📊 应用信息:")
            print(app)
            
    except Exception as e:
        print(f"❌ 应用初始化失败: {e}")
        return 1
    
    # 运行风格演示
    print(f"\n🎯 开始处理 {args.scenes} 个场景...")
    
    all_results = []
    total_start_time = time.time()
    
    for i in range(args.scenes):
        print(f"\n--- 场景 {i+1}/{args.scenes} ---")
        
        try:
            # 运行演示
            demo_result = app.run_style_demo()
            scene_token = demo_result['scene_token']
            
            print(f"✅ 场景 {scene_token[:12]}... 处理完成")
            print(f"  - 地图: {demo_result['scene_metadata'].get('map_name', '未知')}")
            print(f"  - 推理时间: {demo_result['prediction_time']:.2f}秒")
            
            # 保存结果
            viz_path, data_path = save_results(demo_result, output_dir, i+1)
            
            print(f"  - 可视化: {viz_path.name}")
            print(f"  - 数据: {data_path.name}")
            
            # 显示轨迹统计
            if args.verbose:
                print(f"  - 轨迹统计:")
                for style_name, trajectory in demo_result['trajectories'].items():
                    traj_np = trajectory.detach().cpu().numpy()[0]
                    distances = [
                        ((traj_np[i+1, 0] - traj_np[i, 0])**2 + (traj_np[i+1, 1] - traj_np[i, 1])**2)**0.5
                        for i in range(len(traj_np)-1)
                    ]
                    total_dist = sum(distances)
                    print(f"    {style_name}: {total_dist:.2f}米")
            
            all_results.append({
                'scene_index': i+1,
                'scene_token': scene_token,
                'prediction_time': demo_result['prediction_time'],
                'demo_time': demo_result['demo_time'],
                'map_name': demo_result['scene_metadata'].get('map_name', '未知'),
                'files': {
                    'visualization': viz_path.name,
                    'data': data_path.name
                }
            })
            
        except Exception as e:
            print(f"❌ 场景 {i+1} 处理失败: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue
    
    total_time = time.time() - total_start_time
    
    # 保存汇总结果
    summary = {
        'total_scenes': args.scenes,
        'successful_scenes': len(all_results),
        'failed_scenes': args.scenes - len(all_results),
        'total_time': total_time,
        'average_time_per_scene': total_time / len(all_results) if all_results else 0,
        'config': {
            'checkpoint': args.checkpoint,
            'split': args.split,
            'lr': args.lr,
            'seed': args.seed
        },
        'results': all_results
    }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印最终报告
    print(f"\n" + "=" * 60)
    print(f"🎉 处理完成!")
    print(f"  - 总场景: {args.scenes}")
    print(f"  - 成功: {len(all_results)}")
    print(f"  - 失败: {args.scenes - len(all_results)}")
    print(f"  - 总时间: {total_time:.2f}秒")
    print(f"  - 平均时间: {total_time / len(all_results) if all_results else 0:.2f}秒/场景")
    print(f"\n📁 所有结果已保存到: {output_dir.absolute()}")
    print(f"  - 可视化图片: visualizations/")
    print(f"  - 轨迹数据: data/")
    print(f"  - 汇总报告: summary.json")
    
    return 0


def main():
    """主函数"""
    try:
        args = parse_arguments()
        return run_style_demo_cli(args)
    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
        return 1
    except Exception as e:
        print(f"❌ 程序异常: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())