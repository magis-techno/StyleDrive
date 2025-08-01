"""
Evaluation visualization module for PDM results.
Extends navsim.visualization with evaluation-specific functionality.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import asdict

from navsim.common.dataclasses import PDMResults, Trajectory, Frame
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.visualization.config import (
    PDM_RESULTS_CONFIG, 
    TRAJECTORY_CONFIG, 
    TRAJECTORY_COMPARISON_CONFIG,
    BEV_PLOT_CONFIG
)
from navsim.visualization.bev import add_trajectory_to_bev_ax


def _convert_to_trajectory(traj_obj: Any) -> Trajectory:
    """
    Convert different trajectory types to navsim Trajectory format.
    
    :param traj_obj: Trajectory object (could be Trajectory, InterpolatedTrajectory, etc.)
    :return: navsim Trajectory object
    """
    # If already a navsim Trajectory, return as-is
    if isinstance(traj_obj, Trajectory):
        return traj_obj
    
    # Handle InterpolatedTrajectory (nuPlan format)
    if hasattr(traj_obj, '__iter__') and hasattr(traj_obj, '__len__'):
        try:
            # InterpolatedTrajectory is a list of EgoState objects
            ego_states = list(traj_obj)
            if len(ego_states) == 0:
                # Empty trajectory, create minimal one
                poses = np.zeros((1, 3), dtype=np.float32)
            else:
                # Extract poses from EgoState objects
                poses = []
                for ego_state in ego_states:
                    if hasattr(ego_state, 'rear_axle'):
                        # EgoState object
                        pose = ego_state.rear_axle
                        poses.append([pose.x, pose.y, pose.heading])
                    elif hasattr(ego_state, 'center'):
                        # Some other state object
                        pose = ego_state.center
                        poses.append([pose.x, pose.y, pose.heading])
                
                if len(poses) == 0:
                    poses = np.zeros((1, 3), dtype=np.float32)
                else:
                    poses = np.array(poses, dtype=np.float32)
            
            # Create trajectory sampling
            sampling = TrajectorySampling(
                num_poses=len(poses),
                interval_length=0.1  # Default interval
            )
            
            return Trajectory(poses=poses, trajectory_sampling=sampling)
            
        except Exception as e:
            # Fallback: create empty trajectory
            poses = np.zeros((1, 3), dtype=np.float32)
            sampling = TrajectorySampling(num_poses=1, interval_length=0.1)
            return Trajectory(poses=poses, trajectory_sampling=sampling)
    
    # If object has poses attribute, try to use it directly
    if hasattr(traj_obj, 'poses'):
        return traj_obj
    
    # Final fallback: create minimal trajectory
    poses = np.zeros((1, 3), dtype=np.float32)
    sampling = TrajectorySampling(num_poses=1, interval_length=0.1)
    return Trajectory(poses=poses, trajectory_sampling=sampling)


def _get_text_position(position: str, ax: plt.Axes) -> Tuple[float, float]:
    """
    Get text position coordinates based on position string and axis limits.
    
    :param position: Position string ('top-right', 'top-left', 'bottom-right', 'bottom-left')
    :param ax: matplotlib axis object
    :return: (x, y) coordinates for text placement
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    margin_x = (xlim[1] - xlim[0]) * 0.02  # 2% margin
    margin_y = (ylim[1] - ylim[0]) * 0.02  # 2% margin
    
    positions = {
        'top-right': (xlim[1] - margin_x, ylim[1] - margin_y),
        'top-left': (xlim[0] + margin_x, ylim[1] - margin_y),
        'bottom-right': (xlim[1] - margin_x, ylim[0] + margin_y),
        'bottom-left': (xlim[0] + margin_x, ylim[0] + margin_y),
    }
    
    return positions.get(position, positions['top-right'])


def _format_pdm_results_text(pdm_results: PDMResults, style: str = "N") -> str:
    """
    Format PDM results into display text.
    
    :param pdm_results: PDM evaluation results
    :param style: Driving style identifier
    :return: Formatted text string
    """
    config = PDM_RESULTS_CONFIG["metrics_display"]
    decimal_places = config["decimal_places"]
    
    lines = []
    
    # Title
    if config["show_overall_score"]:
        lines.append(f"PDM Score: {pdm_results.score:.{decimal_places}f}")
    
    if config["show_style"]:
        style_name = {"aggressive": "Aggressive", "normal": "Normal", "conservative": "Conservative"}.get(style, f"Style: {style}")
        lines.append(f"Style: {style_name}")
    
    lines.append("")  # Empty line for separation
    
    if config["show_detailed_metrics"]:
        # Safety metrics (multiplicative)
        lines.append("Safety:")
        collision_status = "✓" if pdm_results.no_at_fault_collisions > 0.9 else "✗"
        lines.append(f"  {collision_status} No Collision: {pdm_results.no_at_fault_collisions:.{decimal_places}f}")
        
        drivable_status = "✓" if pdm_results.drivable_area_compliance > 0.9 else "✗"
        lines.append(f"  {drivable_status} Drivable Area: {pdm_results.drivable_area_compliance:.{decimal_places}f}")
        
        lines.append("")
        
        # Efficiency metrics
        lines.append("Efficiency:")
        lines.append(f"  Progress: {pdm_results.ego_progress:.{decimal_places}f}")
        lines.append(f"  Direction: {pdm_results.driving_direction_compliance:.{decimal_places}f}")
        
        lines.append("")
        
        # Comfort metrics
        lines.append("Comfort:")
        lines.append(f"  Comfort: {pdm_results.comfort:.{decimal_places}f}")
        lines.append(f"  TTC: {pdm_results.time_to_collision_within_bound:.{decimal_places}f}")
    
    return "\n".join(lines)


def add_pdm_results_to_bev_ax(
    ax: plt.Axes, 
    pdm_results: PDMResults, 
    style: str = "N",
    position: Optional[str] = None
) -> plt.Axes:
    """
    Add PDM evaluation results as text box to BEV visualization.
    
    :param ax: matplotlib axis object
    :param pdm_results: PDM evaluation results
    :param style: Driving style identifier
    :param position: Text box position ('top-right', 'top-left', 'bottom-right', 'bottom-left')
    :return: Updated axis object
    """
    config = PDM_RESULTS_CONFIG["text_box"]
    position = position or config["position"]
    
    # Format results text
    results_text = _format_pdm_results_text(pdm_results, style)
    
    # Get text position
    x, y = _get_text_position(position, ax)
    
    # Determine text alignment based on position
    ha = 'right' if 'right' in position else 'left'
    va = 'top' if 'top' in position else 'bottom'
    
    # Add text with background box
    text_obj = ax.text(
        x, y, results_text,
        fontsize=config["font_size"],
        fontfamily=config["font_family"],
        color=config["text_color"],
        ha=ha, va=va,
        bbox=config["bbox"],
        zorder=config["zorder"],
        transform=ax.transData
    )
    
    return ax


def add_trajectory_comparison_to_bev_ax(
    ax: plt.Axes,
    trajectories: Dict[str, Trajectory],
    pdm_results: Optional[Dict[str, PDMResults]] = None,
    title: Optional[str] = None
) -> plt.Axes:
    """
    Add multiple trajectories for comparison to BEV visualization.
    
    :param ax: matplotlib axis object
    :param trajectories: Dictionary of trajectory name -> Trajectory object
    :param pdm_results: Optional dictionary of trajectory name -> PDMResults
    :param title: Optional plot title
    :return: Updated axis object
    """
    legend_labels = []
    legend_handles = []
    
    # Plot each trajectory
    for traj_name, trajectory in trajectories.items():
        # Get configuration for this trajectory type
        if traj_name in TRAJECTORY_CONFIG:
            config = TRAJECTORY_CONFIG[traj_name]
        else:
            # Use default agent config for unknown trajectory types
            config = TRAJECTORY_CONFIG["agent"]
        
        # Add trajectory to plot (already converted)
        add_trajectory_to_bev_ax(ax, trajectory, config)
        
        # Create legend entry
        label = traj_name.replace('_', ' ').title()
        if pdm_results and traj_name in pdm_results:
            score = pdm_results[traj_name].score
            label += f" (Score: {score:.3f})"
        
        # Create a line for legend
        line = plt.Line2D(
            [0], [0],
            color=config["line_color"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            marker=config["marker"],
            markersize=config["marker_size"],
            alpha=config["line_color_alpha"]
        )
        
        legend_handles.append(line)
        legend_labels.append(label)
    
    # Add legend if configured
    legend_config = TRAJECTORY_COMPARISON_CONFIG["legend"]
    if legend_config["show"] and legend_labels:
        ax.legend(
            legend_handles, legend_labels,
            loc=legend_config["location"],
            fontsize=legend_config["font_size"],
            framealpha=legend_config["frame_alpha"],
            fancybox=legend_config["fancybox"],
            shadow=legend_config["shadow"]
        )
    
    # Add title if configured
    title_config = TRAJECTORY_COMPARISON_CONFIG["title"]
    if title and title_config["show"]:
        ax.set_title(
            title,
            fontsize=title_config["font_size"],
            fontweight=title_config["font_weight"]
        )
    
    return ax


def create_evaluation_visualization(
    frame: Frame,
    trajectories: Dict[str, Trajectory],
    pdm_results: Dict[str, PDMResults],
    map_api: Optional[Any] = None,
    scene_token: Optional[str] = None,
    style: str = "N"
) -> plt.Figure:
    """
    Create a complete evaluation visualization with BEV background and PDM results.
    
    :param frame: NavSim frame containing ego state and annotations
    :param trajectories: Dictionary of trajectory name -> Trajectory object
    :param pdm_results: Dictionary of trajectory name -> PDMResults
    :param map_api: Optional map API for background rendering
    :param scene_token: Optional scene token for title
    :param style: Driving style identifier
    :return: matplotlib Figure object
    """
    from navsim.visualization.bev import add_configured_bev_on_ax
    from navsim.visualization.plots import configure_bev_ax
    
    # Convert all trajectories to consistent format at the beginning
    converted_trajectories = {}
    for traj_name, trajectory in trajectories.items():
        try:
            converted_trajectories[traj_name] = _convert_to_trajectory(trajectory)
        except Exception as e:
            # Log warning and skip this trajectory
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to convert trajectory {traj_name}: {e}")
    
    # Create figure and axis
    fig_size = BEV_PLOT_CONFIG["figure_size"]
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    fig.patch.set_facecolor(BEV_PLOT_CONFIG["background_color"])
    
    # Configure BEV axis
    ax = configure_bev_ax(ax)
    
    # Add BEV background (map + annotations)
    if map_api:
        ax = add_configured_bev_on_ax(ax, map_api, frame)
    else:
        # Add at least annotations if no map available
        from navsim.visualization.bev import add_annotations_to_bev_ax
        ax = add_annotations_to_bev_ax(ax, frame.annotations)
    
    # Add trajectory comparison
    title = None
    if scene_token:
        title = f"Scene: {scene_token[:8]}... | Style: {style}"
    
    ax = add_trajectory_comparison_to_bev_ax(ax, converted_trajectories, pdm_results, title)
    
    # Add PDM results for the predicted trajectory
    if "predicted" in pdm_results:
        ax = add_pdm_results_to_bev_ax(ax, pdm_results["predicted"], style)
    elif pdm_results:
        # If no "predicted" key, use the first available result
        first_key = next(iter(pdm_results.keys()))
        ax = add_pdm_results_to_bev_ax(ax, pdm_results[first_key], style)
    
    plt.tight_layout()
    return fig


def save_evaluation_results(
    evaluation_results: Dict[str, Any],
    output_dir: str,
    scene_token: str,
    file_format: str = "png"
) -> str:
    """
    Save evaluation results including visualization and data.
    
    :param evaluation_results: Dictionary containing figure, trajectories, and PDM results
    :param output_dir: Output directory path
    :param scene_token: Scene token for filename
    :param file_format: Image file format (png, jpg, pdf)
    :return: Path to saved visualization file
    """
    import os
    import json
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save visualization
    fig = evaluation_results["figure"]
    viz_filename = f"{scene_token[:12]}_evaluation.{file_format}"
    viz_path = os.path.join(output_dir, viz_filename)
    fig.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Save detailed data
    data_filename = f"{scene_token[:12]}_evaluation_data.json"
    data_path = os.path.join(output_dir, data_filename)
    
    # Convert PDM results to serializable format
    serializable_results = {}
    for key, pdm_result in evaluation_results["pdm_results"].items():
        serializable_results[key] = asdict(pdm_result)
    
    data = {
        "scene_token": scene_token,
        "style": evaluation_results.get("style", "N"),
        "pdm_results": serializable_results,
        "trajectory_info": {
            name: {
                "num_poses": len(traj.poses),
                "sampling_time": traj.trajectory_sampling.time_horizon,
                "interval_length": traj.trajectory_sampling.interval_length
            }
            for name, traj in evaluation_results["trajectories"].items()
        }
    }
    
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Optionally save trajectories as pickle for later analysis
    traj_filename = f"{scene_token[:12]}_trajectories.pkl"
    traj_path = os.path.join(output_dir, traj_filename)
    with open(traj_path, 'wb') as f:
        pickle.dump(evaluation_results["trajectories"], f)
    
    return viz_path