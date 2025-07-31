"""
Style Visualization Module

This module provides visualization functions for comparing trajectories
from different driving styles in BEV (Bird's Eye View) format.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Any
import torch

# Try to import navsim visualization modules
try:
    from navsim.visualization.bev import plot_bev_map
    from navsim.visualization.plots import plot_trajectory
    NAVSIM_VIZ_AVAILABLE = True
except ImportError:
    NAVSIM_VIZ_AVAILABLE = False
    print("Warning: navsim visualization modules not available. Using simplified plotting.")


def plot_style_trajectories(
    trajectories: Dict[str, torch.Tensor], 
    scene_data: Optional[Any] = None,
    title: str = "Driving Style Trajectory Comparison",
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot trajectories from different driving styles in a single BEV view.
    
    Args:
        trajectories: Dictionary mapping style names to trajectory tensors
        scene_data: Optional scene data for background visualization
        title: Plot title
        figsize: Figure size tuple
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    # Style colors consistent with inference class
    style_colors = {
        'aggressive': 'red',
        'normal': 'blue', 
        'conservative': 'green'
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot background scene if available
    if scene_data is not None and NAVSIM_VIZ_AVAILABLE:
        # TODO: Implement scene background plotting using navsim
        pass
    
    # Plot each style trajectory
    for style_name, trajectory in trajectories.items():
        color = style_colors.get(style_name, 'black')
        
        # Convert tensor to numpy if needed
        if isinstance(trajectory, torch.Tensor):
            traj_np = trajectory.detach().cpu().numpy()
        else:
            traj_np = trajectory
            
        # Handle different trajectory formats
        if len(traj_np.shape) == 3:  # [batch, timesteps, dims]
            traj_np = traj_np[0]  # Take first batch
            
        # Extract x, y coordinates (assuming first 2 dimensions are x, y)
        x_coords = traj_np[:, 0]
        y_coords = traj_np[:, 1]
        
        # Plot trajectory
        ax.plot(x_coords, y_coords, 
               color=color, linewidth=3, alpha=0.8,
               label=f'{style_name.capitalize()} Style',
               marker='o', markersize=4)
        
        # Mark start point
        ax.scatter(x_coords[0], y_coords[0], 
                  color=color, s=100, marker='s', 
                  edgecolor='black', linewidth=2,
                  label=f'{style_name.capitalize()} Start')
        
        # Mark end point  
        ax.scatter(x_coords[-1], y_coords[-1],
                  color=color, s=100, marker='^',
                  edgecolor='black', linewidth=2,
                  label=f'{style_name.capitalize()} End')
    
    # Formatting
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def plot_trajectory_metrics(trajectories: Dict[str, torch.Tensor]) -> plt.Figure:
    """
    Plot quantitative metrics comparing different style trajectories.
    
    Args:
        trajectories: Dictionary mapping style names to trajectory tensors
        
    Returns:
        matplotlib Figure object
    """
    metrics = {}
    
    for style_name, trajectory in trajectories.items():
        # Convert to numpy
        if isinstance(trajectory, torch.Tensor):
            traj_np = trajectory.detach().cpu().numpy()
        else:
            traj_np = trajectory
            
        if len(traj_np.shape) == 3:
            traj_np = traj_np[0]  # Take first batch
            
        # Calculate metrics
        x_coords = traj_np[:, 0]
        y_coords = traj_np[:, 1]
        
        # Total distance
        distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
        total_distance = np.sum(distances)
        
        # Average speed (assuming time steps are uniform)
        avg_speed = np.mean(distances)
        
        # Max deviation from straight line
        start_point = np.array([x_coords[0], y_coords[0]])
        end_point = np.array([x_coords[-1], y_coords[-1]])
        line_vec = end_point - start_point
        line_length = np.linalg.norm(line_vec)
        
        if line_length > 0:
            line_unit = line_vec / line_length
            max_deviation = 0
            for i in range(len(x_coords)):
                point = np.array([x_coords[i], y_coords[i]])
                point_vec = point - start_point
                projection = np.dot(point_vec, line_unit) * line_unit
                deviation = np.linalg.norm(point_vec - projection)
                max_deviation = max(max_deviation, deviation)
        else:
            max_deviation = 0
            
        metrics[style_name] = {
            'total_distance': total_distance,
            'avg_speed': avg_speed,
            'max_deviation': max_deviation
        }
    
    # Plot metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    styles = list(metrics.keys())
    colors = ['red', 'blue', 'green']
    
    # Total distance
    distances = [metrics[style]['total_distance'] for style in styles]
    axes[0].bar(styles, distances, color=colors)
    axes[0].set_title('Total Distance')
    axes[0].set_ylabel('Distance (m)')
    
    # Average speed
    speeds = [metrics[style]['avg_speed'] for style in styles]
    axes[1].bar(styles, speeds, color=colors)
    axes[1].set_title('Average Speed')
    axes[1].set_ylabel('Speed (m/timestep)')
    
    # Max deviation
    deviations = [metrics[style]['max_deviation'] for style in styles]
    axes[2].bar(styles, deviations, color=colors)
    axes[2].set_title('Max Deviation from Straight Line')
    axes[2].set_ylabel('Deviation (m)')
    
    plt.tight_layout()
    return fig


def create_style_comparison_grid(trajectories: Dict[str, torch.Tensor], scene_data=None) -> plt.Figure:
    """
    Create a grid showing individual and combined trajectory plots.
    
    Args:
        trajectories: Dictionary mapping style names to trajectory tensors
        scene_data: Optional scene data for background
        
    Returns:
        matplotlib Figure object with subplot grid
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    style_colors = {
        'aggressive': 'red',
        'normal': 'blue', 
        'conservative': 'green'
    }
    
    # Individual style plots
    for idx, (style_name, trajectory) in enumerate(trajectories.items()):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        
        # Convert tensor to numpy
        if isinstance(trajectory, torch.Tensor):
            traj_np = trajectory.detach().cpu().numpy()
        else:
            traj_np = trajectory
            
        if len(traj_np.shape) == 3:
            traj_np = traj_np[0]
            
        x_coords = traj_np[:, 0]
        y_coords = traj_np[:, 1]
        color = style_colors[style_name]
        
        ax.plot(x_coords, y_coords, color=color, linewidth=4, alpha=0.8)
        ax.scatter(x_coords[0], y_coords[0], color=color, s=100, marker='s', edgecolor='black')
        ax.scatter(x_coords[-1], y_coords[-1], color=color, s=100, marker='^', edgecolor='black')
        
        ax.set_title(f'{style_name.capitalize()} Style', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Combined plot in the last subplot
    ax = axes[1, 1]
    for style_name, trajectory in trajectories.items():
        if isinstance(trajectory, torch.Tensor):
            traj_np = trajectory.detach().cpu().numpy()
        else:
            traj_np = trajectory
            
        if len(traj_np.shape) == 3:
            traj_np = traj_np[0]
            
        x_coords = traj_np[:, 0]
        y_coords = traj_np[:, 1]
        color = style_colors[style_name]
        
        ax.plot(x_coords, y_coords, color=color, linewidth=3, alpha=0.8, 
               label=f'{style_name.capitalize()}')
        ax.scatter(x_coords[0], y_coords[0], color=color, s=80, marker='s', edgecolor='black')
        ax.scatter(x_coords[-1], y_coords[-1], color=color, s=80, marker='^', edgecolor='black')
    
    ax.set_title('All Styles Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig