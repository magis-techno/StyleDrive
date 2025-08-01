"""
StyleDrive Trajectory App

Complete application framework for DiffusionDrive-Style model.
Supports style-aware trajectory prediction, data management, and visualization.
"""

__version__ = "0.2.0"
__author__ = "StyleDrive Team"

# Import main application class
from .app import StyleTrajectoryApp

# Import core components
from .inference_engine import StyleTrajectoryInferenceEngine
from .data_manager import StyleTrajectoryDataManager

# Import visualization functions (keep backward compatibility)
from .style_visualization import (
    plot_style_trajectories,
    plot_style_trajectories_bev,
    plot_trajectory_metrics,
    create_style_comparison_grid
)

# Import legacy classes for backward compatibility
from .diffusion_style_inference import DiffusionStyleInference

# Import configuration utilities
from .config import load_default_config, load_config

__all__ = [
    # Main application
    "StyleTrajectoryApp",
    
    # Core components
    "StyleTrajectoryInferenceEngine", 
    "StyleTrajectoryDataManager",
    
    # Visualization functions
    "plot_style_trajectories",
    "plot_style_trajectories_bev",
    "plot_trajectory_metrics", 
    "create_style_comparison_grid",
    
    # Legacy compatibility
    "DiffusionStyleInference",
    
    # Configuration
    "load_default_config",
    "load_config"
]