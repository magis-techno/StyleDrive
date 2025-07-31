"""
StyleTrajectoryApp package entry point for command line usage.

Usage:
    python -m style_trajectory_app --checkpoint model.ckpt --dataset /data/navsim
"""

from .cli import main

if __name__ == '__main__':
    main()