"""
Multi-style evaluation visualization for comparing different driving styles.
"""

import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from navsim.common.dataclasses import PDMResults, Trajectory, Frame
from navsim.visualization.evaluation_viz import (
    _convert_to_trajectory, 
    add_pdm_results_to_bev_ax,
    _format_pdm_results_text
)
from navsim.visualization.config import (
    TRAJECTORY_CONFIG, 
    BEV_PLOT_CONFIG,
    PDM_RESULTS_CONFIG
)
from navsim.visualization.bev import add_trajectory_to_bev_ax
from navsim.visualization.plots import configure_bev_ax


def create_multi_style_comparison(
    frame: Frame,
    style_results: Dict[str, Dict[str, Any]],
    map_api: Optional[Any] = None,
    scene_token: Optional[str] = None,
    gt_trajectory: Optional[Trajectory] = None
) -> plt.Figure:
    """
    Create visualization comparing multiple driving styles for the same scenario.
    
    :param frame: NavSim frame containing ego state and annotations
    :param style_results: Dictionary of {style_name: {"trajectory": traj, "pdm_result": result}}
    :param map_api: Optional map API for background rendering
    :param scene_token: Optional scene token for title
    :param gt_trajectory: Optional ground truth trajectory
    :return: matplotlib Figure object
    """
    from navsim.visualization.bev import add_configured_bev_on_ax, add_annotations_to_bev_ax
    
    # Create figure with subplots
    num_styles = len(style_results)
    if num_styles <= 3:
        fig, axes = plt.subplots(1, num_styles + 1, figsize=(5 * (num_styles + 1), 5))
    else:
        rows = (num_styles + 2) // 3  # +1 for overview, +1 for ceiling division
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        axes = axes.flatten()
    
    if num_styles == 1:
        axes = [axes]
    
    fig.patch.set_facecolor(BEV_PLOT_CONFIG["background_color"])
    
    # Style colors for different trajectories
    style_colors = {
        "aggressive": "#d62728",    # red
        "normal": "#2ca02c",        # green  
        "conservative": "#1f77b4",  # blue
    }
    
    # First subplot: Overview with all styles
    ax_overview = axes[0]
    ax_overview = configure_bev_ax(ax_overview)
    
    # Add background
    if map_api:
        ax_overview = add_configured_bev_on_ax(ax_overview, map_api, frame)
    else:
        ax_overview = add_annotations_to_bev_ax(ax_overview, frame.annotations)
    
    # Add ground truth if available
    if gt_trajectory:
        gt_converted = _convert_to_trajectory(gt_trajectory)
        add_trajectory_to_bev_ax(ax_overview, gt_converted, TRAJECTORY_CONFIG["ground_truth"])
    
    # Add all style trajectories to overview
    legend_handles = []
    legend_labels = []
    
    for style_name, result_data in style_results.items():
        trajectory = _convert_to_trajectory(result_data["trajectory"])
        
        # Create custom config for this style
        color = style_colors.get(style_name, "#ff7f0e")
        style_config = {
            "line_color": color,
            "line_color_alpha": 0.8,
            "line_width": 2.5,
            "line_style": "-",
            "marker": "s",
            "marker_size": 4,
            "marker_edge_color": "black",
            "zorder": 4,
        }
        
        add_trajectory_to_bev_ax(ax_overview, trajectory, style_config)
        
        # Create legend entry
        line = plt.Line2D([0], [0], color=color, linewidth=2.5, marker='s', markersize=4)
        legend_handles.append(line)
        
        # Add score to legend
        if "pdm_result" in result_data:
            score = result_data["pdm_result"].score
            legend_labels.append(f"{style_name.title()} (Score: {score:.3f})")
        else:
            legend_labels.append(style_name.title())
    
    # Add GT to legend if available
    if gt_trajectory:
        gt_line = plt.Line2D([0], [0], color=TRAJECTORY_CONFIG["ground_truth"]["line_color"], 
                            linewidth=2.5, marker='o', markersize=4)
        legend_handles.append(gt_line)
        legend_labels.append("Ground Truth")
    
    ax_overview.legend(legend_handles, legend_labels, loc='upper left', fontsize=9)
    ax_overview.set_title(f"Multi-Style Comparison\nScene: {scene_token[:8] if scene_token else 'Unknown'}", 
                         fontsize=12, fontweight='bold')
    
    # Individual style subplots
    for idx, (style_name, result_data) in enumerate(style_results.items()):
        ax = axes[idx + 1]
        ax = configure_bev_ax(ax)
        
        # Add background
        if map_api:
            ax = add_configured_bev_on_ax(ax, map_api, frame)
        else:
            ax = add_annotations_to_bev_ax(ax, frame.annotations)
        
        # Add trajectories
        if gt_trajectory:
            gt_converted = _convert_to_trajectory(gt_trajectory)
            add_trajectory_to_bev_ax(ax, gt_converted, TRAJECTORY_CONFIG["ground_truth"])
        
        trajectory = _convert_to_trajectory(result_data["trajectory"])
        color = style_colors.get(style_name, "#ff7f0e")
        style_config = {
            "line_color": color,
            "line_color_alpha": 1.0,
            "line_width": 3.0,
            "line_style": "-",
            "marker": "s",
            "marker_size": 5,
            "marker_edge_color": "black",
            "zorder": 4,
        }
        add_trajectory_to_bev_ax(ax, trajectory, style_config)
        
        # Add PDM results if available
        if "pdm_result" in result_data:
            ax = add_pdm_results_to_bev_ax(ax, result_data["pdm_result"], style_name, position="top-right")
        
        ax.set_title(f"{style_name.title()} Style", fontsize=11, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(num_styles + 1, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def run_multi_style_evaluation(
    agent_input: Any,
    scene: Any,
    agent: Any,
    styles: List[str] = ["aggressive", "normal", "conservative"],
    simulator: Optional[Any] = None,
    scorer: Optional[Any] = None,
    metric_cache: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Run evaluation for multiple driving styles on the same scenario.
    
    :param agent_input: Agent input for the scenario
    :param scene: Scene object
    :param agent: Style-aware agent
    :param styles: List of styles to evaluate
    :param simulator: PDM simulator (optional)
    :param scorer: PDM scorer (optional)
    :param metric_cache: Metric cache (optional)
    :return: Dictionary containing results for each style
    """
    from navsim.evaluate.pdm_score import pdm_score
    
    results = {}
    gt_trajectory = scene.get_future_trajectory()
    
    for style in styles:
        try:
            # Generate trajectory for this style
            # Note: This assumes the agent supports style input
            # You may need to modify this based on your agent's interface
            if hasattr(agent, 'compute_trajectory_with_style'):
                trajectory = agent.compute_trajectory_with_style(agent_input, style)
            else:
                # Fallback: use regular compute_trajectory
                # You might need to set the style in the agent beforehand
                trajectory = agent.compute_trajectory(agent_input, scene.frames[3].token)
            
            # Run PDM evaluation if components are available
            pdm_result = None
            if all([simulator, scorer, metric_cache]):
                pdm_result = pdm_score(
                    metric_cache=metric_cache,
                    model_trajectory=trajectory,
                    future_sampling=simulator.proposal_sampling,
                    simulator=simulator,
                    scorer=scorer,
                    style=style,
                    gt_trajectory=gt_trajectory
                )
            
            results[style] = {
                "trajectory": trajectory,
                "pdm_result": pdm_result
            }
            
        except Exception as e:
            print(f"Failed to evaluate style {style}: {e}")
            continue
    
    return {
        "style_results": results,
        "ground_truth": gt_trajectory,
        "scene": scene
    }


def save_multi_style_visualization(
    evaluation_results: Dict[str, Any],
    output_dir: str,
    scene_token: str
) -> str:
    """
    Save multi-style comparison visualization.
    
    :param evaluation_results: Results from run_multi_style_evaluation
    :param output_dir: Output directory
    :param scene_token: Scene token
    :return: Path to saved visualization
    """
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    scene = evaluation_results["scene"]
    current_frame = scene.frames[0]
    
    try:
        map_api = scene.map_api
    except:
        map_api = None
    
    fig = create_multi_style_comparison(
        frame=current_frame,
        style_results=evaluation_results["style_results"],
        map_api=map_api,
        scene_token=scene_token,
        gt_trajectory=evaluation_results["ground_truth"]
    )
    
    # Save visualization
    viz_filename = f"{scene_token[:12]}_multi_style_comparison.png"
    viz_path = os.path.join(output_dir, viz_filename)
    fig.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Save data
    data_filename = f"{scene_token[:12]}_multi_style_data.json"
    data_path = os.path.join(output_dir, data_filename)
    
    # Convert results to serializable format
    serializable_data = {
        "scene_token": scene_token,
        "styles": {}
    }
    
    for style_name, result_data in evaluation_results["style_results"].items():
        style_data = {"style": style_name}
        if result_data["pdm_result"]:
            from dataclasses import asdict
            style_data["pdm_metrics"] = asdict(result_data["pdm_result"])
        serializable_data["styles"][style_name] = style_data
    
    with open(data_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    plt.close(fig)
    return viz_path