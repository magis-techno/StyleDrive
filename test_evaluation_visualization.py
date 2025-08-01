#!/usr/bin/env python3
"""
Test script for evaluation visualization functionality.
Creates synthetic data to test the visualization pipeline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List

# Import navsim components
from navsim.common.dataclasses import PDMResults, Trajectory, Frame, Annotations
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import TrajectorySampling
from navsim.visualization.evaluation_viz import (
    add_pdm_results_to_bev_ax,
    add_trajectory_comparison_to_bev_ax,
    create_evaluation_visualization,
    save_evaluation_results
)
from navsim.visualization.plots import configure_bev_ax


def create_synthetic_trajectory(
    start_pos: List[float] = [0.0, 0.0],
    end_pos: List[float] = [30.0, 5.0],
    num_points: int = 20,
    style: str = "normal"
) -> Trajectory:
    """Create a synthetic trajectory for testing."""
    
    # Create trajectory points based on style
    if style == "aggressive":
        # More curved, faster changes
        x_points = np.linspace(start_pos[0], end_pos[0], num_points)
        y_points = start_pos[1] + (end_pos[1] - start_pos[1]) * np.sin(np.linspace(0, np.pi, num_points)) * 1.5
        headings = np.linspace(0, 0.3, num_points)
    elif style == "conservative":
        # Straighter, more gradual
        x_points = np.linspace(start_pos[0], end_pos[0], num_points)
        y_points = np.linspace(start_pos[1], end_pos[1], num_points)
        headings = np.linspace(0, 0.1, num_points)
    else:  # normal
        # Moderate curve
        x_points = np.linspace(start_pos[0], end_pos[0], num_points)
        y_points = start_pos[1] + (end_pos[1] - start_pos[1]) * np.linspace(0, 1, num_points) + \
                   np.sin(np.linspace(0, np.pi/2, num_points)) * 2
        headings = np.linspace(0, 0.2, num_points)
    
    # Create poses array [x, y, heading]
    poses = np.column_stack([x_points, y_points, headings])
    
    # Create trajectory sampling
    trajectory_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.2)
    
    return Trajectory(poses=poses, trajectory_sampling=trajectory_sampling)


def create_synthetic_pdm_results(style: str = "normal") -> PDMResults:
    """Create synthetic PDM results for testing."""
    
    # Vary results based on style
    if style == "aggressive":
        return PDMResults(
            no_at_fault_collisions=0.95,
            drivable_area_compliance=0.88,
            ego_progress=0.92,
            time_to_collision_within_bound=0.78,
            comfort=0.65,
            driving_direction_compliance=0.85,
            score=0.82
        )
    elif style == "conservative":
        return PDMResults(
            no_at_fault_collisions=1.0,
            drivable_area_compliance=0.98,
            ego_progress=0.78,
            time_to_collision_within_bound=0.95,
            comfort=0.92,
            driving_direction_compliance=0.95,
            score=0.91
        )
    else:  # normal
        return PDMResults(
            no_at_fault_collisions=0.98,
            drivable_area_compliance=0.92,
            ego_progress=0.85,
            time_to_collision_within_bound=0.88,
            comfort=0.82,
            driving_direction_compliance=0.90,
            score=0.87
        )


def create_synthetic_frame() -> Frame:
    """Create a synthetic frame with annotations for testing."""
    
    # Create some vehicle annotations around the ego vehicle
    # Format: [x, y, heading, length, width, height]
    boxes = np.array([
        [10.0, 2.0, 0.1, 4.5, 2.0, 1.8],   # Vehicle ahead-right
        [8.0, -3.0, 0.0, 4.2, 1.9, 1.7],   # Vehicle ahead-left
        [-5.0, 1.0, 0.0, 4.0, 1.8, 1.6],   # Vehicle behind
    ])
    
    # Vehicle type indices (assuming VEHICLE = 0)
    names = np.array([0, 0, 0])
    
    annotations = Annotations(names=names, boxes=boxes)
    
    # Create a minimal frame (we'll skip lidar and cameras for this test)
    frame = Frame(
        ego_status=None,  # Will use default ego at origin
        annotations=annotations,
        lidar=None,
        cameras=[]
    )
    
    return frame


def test_basic_visualization():
    """Test basic visualization functions."""
    print("Testing basic visualization functions...")
    
    # Create test data
    pdm_results = create_synthetic_pdm_results("normal")
    
    # Test 1: PDM results text box
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax = configure_bev_ax(ax)
    ax = add_pdm_results_to_bev_ax(ax, pdm_results, "normal")
    
    plt.title("Test 1: PDM Results Display")
    plt.savefig("test_pdm_results_display.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ PDM results display test completed")
    
    # Test 2: Multiple trajectory comparison
    trajectories = {
        "predicted": create_synthetic_trajectory([0, 0], [30, 5], style="normal"),
        "ground_truth": create_synthetic_trajectory([0, 0], [30, 3], style="conservative"),
        "pdm_reference": create_synthetic_trajectory([0, 0], [30, 4], style="aggressive")
    }
    
    pdm_results_dict = {
        "predicted": create_synthetic_pdm_results("normal"),
        "ground_truth": create_synthetic_pdm_results("conservative"),
        "pdm_reference": create_synthetic_pdm_results("aggressive")
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax = configure_bev_ax(ax)
    ax = add_trajectory_comparison_to_bev_ax(
        ax, trajectories, pdm_results_dict, 
        title="Test Trajectory Comparison"
    )
    
    plt.savefig("test_trajectory_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Trajectory comparison test completed")


def test_complete_visualization():
    """Test complete evaluation visualization."""
    print("Testing complete evaluation visualization...")
    
    # Create synthetic data
    frame = create_synthetic_frame()
    trajectories = {
        "predicted": create_synthetic_trajectory([0, 0], [30, 5], style="aggressive"),
        "ground_truth": create_synthetic_trajectory([0, 0], [30, 2], style="normal"),
        "pdm_reference": create_synthetic_trajectory([0, 0], [30, 3], style="conservative")
    }
    
    pdm_results = {
        "predicted": create_synthetic_pdm_results("aggressive"),
        "ground_truth": create_synthetic_pdm_results("normal"),
        "pdm_reference": create_synthetic_pdm_results("conservative")
    }
    
    # Create complete visualization (without map API)
    fig = create_evaluation_visualization(
        frame=frame,
        trajectories=trajectories,
        pdm_results=pdm_results,
        map_api=None,  # No map for this test
        scene_token="test_scene_12345",
        style="aggressive"
    )
    
    plt.savefig("test_complete_evaluation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Complete evaluation visualization test completed")


def test_coordinate_system():
    """Test coordinate system to ensure correct rendering."""
    print("Testing coordinate system...")
    
    # Create a simple trajectory that moves forward (positive x) and left (positive y)
    # According to user: "向上是x的正方向，向左是y的正方向"
    test_points = np.array([
        [0, 0, 0],      # Start at origin
        [10, 0, 0],     # Move forward (up in visualization)
        [20, 5, 0.1],   # Move forward and left
        [30, 10, 0.2],  # Continue forward and left
    ])
    
    trajectory = Trajectory(
        poses=test_points,
        trajectory_sampling=TrajectorySampling(time_horizon=3.0, interval_length=1.0)
    )
    
    pdm_results = create_synthetic_pdm_results("normal")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    ax = configure_bev_ax(ax)
    
    # Add trajectory
    from navsim.visualization.bev import add_trajectory_to_bev_ax
    from navsim.visualization.config import TRAJECTORY_CONFIG
    add_trajectory_to_bev_ax(ax, trajectory, TRAJECTORY_CONFIG["predicted"])
    
    # Add coordinate reference
    ax.annotate("Forward (+X)", xy=(0, 15), xytext=(0, 20), 
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                fontsize=12, ha='center', color='blue')
    ax.annotate("Left (+Y)", xy=(-10, 0), xytext=(-15, 0),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=12, ha='center', color='green', rotation=90)
    
    # Mark start and end points
    ax.plot(0, 0, 'ro', markersize=10, label='Start (0,0)')
    ax.plot(test_points[-1, 1], test_points[-1, 0], 'go', markersize=10, label='End (30,10)')
    
    # Add PDM results
    add_pdm_results_to_bev_ax(ax, pdm_results, "normal", position="bottom-right")
    
    ax.legend()
    plt.title("Coordinate System Test\n(向上是x的正方向，向左是y的正方向)")
    plt.savefig("test_coordinate_system.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Coordinate system test completed")


def test_style_comparison():
    """Test visualization with different driving styles."""
    print("Testing style comparison...")
    
    styles = ["aggressive", "normal", "conservative"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, style in enumerate(styles):
        ax = axes[i]
        ax = configure_bev_ax(ax)
        
        # Create trajectory and results for this style
        trajectory = create_synthetic_trajectory([0, 0], [30, 5], style=style)
        pdm_results = create_synthetic_pdm_results(style)
        
        # Add trajectory
        from navsim.visualization.bev import add_trajectory_to_bev_ax
        from navsim.visualization.config import TRAJECTORY_CONFIG
        add_trajectory_to_bev_ax(ax, trajectory, TRAJECTORY_CONFIG["predicted"])
        
        # Add results
        add_pdm_results_to_bev_ax(ax, pdm_results, style)
        
        ax.set_title(f"{style.title()} Style")
    
    plt.tight_layout()
    plt.savefig("test_style_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Style comparison test completed")


def main():
    """Run all visualization tests."""
    print("🚗 Testing NavSim Evaluation Visualization")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("test_outputs", exist_ok=True)
    os.chdir("test_outputs")
    
    try:
        # Run all tests
        test_basic_visualization()
        test_complete_visualization() 
        test_coordinate_system()
        test_style_comparison()
        
        print("\n✅ All visualization tests completed successfully!")
        print(f"📁 Test outputs saved in: {os.getcwd()}")
        print("\nGenerated files:")
        for file in os.listdir("."):
            if file.endswith(".png"):
                print(f"  - {file}")
                
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())