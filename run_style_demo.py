#!/usr/bin/env python3
"""
StyleTrajectoryApp 快速启动脚本

直接运行风格演示的便捷脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from style_trajectory_app.cli import main

if __name__ == '__main__':
    main()