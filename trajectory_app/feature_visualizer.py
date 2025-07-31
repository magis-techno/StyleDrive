"""
Feature Visualizer for DiffusionDrive Model

This module provides visualization capabilities for intermediate features
extracted from the DiffusionDrive model, including BEV semantic segmentation,
attention weights, and other neural network features.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import cv2
from typing import Dict, Any, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class FeatureVisualizer:
    """
    Visualizer for DiffusionDrive model features
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature visualizer
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Define semantic classes based on DiffusionDrive config
        self.bev_semantic_classes = {
            0: {"name": "background", "color": "#000000", "type": "background"},
            1: {"name": "road", "color": "#808080", "type": "polygon"},  # Gray for roads
            2: {"name": "walkways", "color": "#8B4513", "type": "polygon"},  # Brown for walkways  
            3: {"name": "centerline", "color": "#FFFF00", "type": "linestring"},  # Yellow for centerlines
            4: {"name": "static_objects", "color": "#FF0000", "type": "box"},  # Red for static objects
            5: {"name": "vehicles", "color": "#0000FF", "type": "box"},  # Blue for vehicles
            6: {"name": "pedestrians", "color": "#00FF00", "type": "box"},  # Green for pedestrians
        }
        
        # Create colormap for semantic segmentation
        self.semantic_colormap = self._create_semantic_colormap()
        
    def _create_semantic_colormap(self) -> ListedColormap:
        """
        Create a colormap for semantic segmentation visualization
        
        Returns:
            Matplotlib colormap
        """
        colors = []
        for class_id in sorted(self.bev_semantic_classes.keys()):
            hex_color = self.bev_semantic_classes[class_id]["color"]
            # Convert hex to RGB (0-1 range)
            rgb = mcolors.hex2color(hex_color)
            colors.append(rgb)
            
        return ListedColormap(colors)
    
    def visualize_bev_semantic_map(
        self, 
        semantic_map: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
        overlay_alpha: float = 0.7,
        show_legend: bool = True,
        title: str = "BEV Semantic Segmentation"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize BEV semantic segmentation map
        
        Args:
            semantic_map: [H, W] array of class predictions
            confidence_map: Optional [H, W] array of prediction confidence
            overlay_alpha: Alpha value for overlay visualization
            show_legend: Whether to show legend
            title: Plot title
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, axes = plt.subplots(1, 2 if confidence_map is not None else 1, 
                                figsize=(15, 7) if confidence_map is not None else (8, 7))
        
        if confidence_map is not None:
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            ax_semantic, ax_confidence = axes
        else:
            ax_semantic = axes if not isinstance(axes, np.ndarray) else axes[0]
            
        # Main semantic segmentation visualization
        im = ax_semantic.imshow(
            semantic_map, 
            cmap=self.semantic_colormap, 
            vmin=0, 
            vmax=len(self.bev_semantic_classes)-1,
            alpha=overlay_alpha,
            origin='lower'  # Match BEV coordinate convention
        )
        
        ax_semantic.set_title(title)
        ax_semantic.set_xlabel("BEV X (sideways)")
        ax_semantic.set_ylabel("BEV Y (forward)")
        
        # Add legend if requested
        if show_legend:
            self._add_semantic_legend(ax_semantic, semantic_map)
            
        # Confidence visualization if provided
        if confidence_map is not None:
            conf_im = ax_confidence.imshow(
                confidence_map,
                cmap='viridis',
                alpha=overlay_alpha,
                origin='lower'
            )
            ax_confidence.set_title("Prediction Confidence")
            ax_confidence.set_xlabel("BEV X (sideways)")
            ax_confidence.set_ylabel("BEV Y (forward)")
            
            # Add colorbar for confidence
            plt.colorbar(conf_im, ax=ax_confidence, label="Confidence")
            
        plt.tight_layout()
        return fig, axes
    
    def _add_semantic_legend(self, ax: plt.Axes, semantic_map: np.ndarray) -> None:
        """
        Add legend for semantic classes
        
        Args:
            ax: Matplotlib axes
            semantic_map: Semantic segmentation map to check which classes are present
        """
        # Only show legend for classes that are actually present in the map
        present_classes = np.unique(semantic_map)
        
        legend_elements = []
        for class_id in present_classes:
            if class_id in self.bev_semantic_classes:
                class_info = self.bev_semantic_classes[class_id]
                legend_elements.append(
                    plt.Rectangle((0, 0), 1, 1, 
                                facecolor=class_info["color"], 
                                label=f"{class_id}: {class_info['name']}")
                )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    def create_semantic_overlay(
        self,
        background_image: np.ndarray,
        semantic_map: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create overlay of semantic segmentation on background image
        
        Args:
            background_image: Background image [H, W, 3] in RGB
            semantic_map: Semantic segmentation [H, W] 
            alpha: Overlay transparency
            
        Returns:
            Overlaid image [H, W, 3]
        """
        # Ensure background is RGB and float
        if background_image.dtype != np.float32:
            background_image = background_image.astype(np.float32) / 255.0
            
        # Create semantic RGB image
        semantic_rgb = self.semantic_colormap(semantic_map)[:, :, :3]  # Remove alpha channel
        
        # Blend images
        overlaid = (1 - alpha) * background_image + alpha * semantic_rgb
        
        # Convert back to uint8
        return (overlaid * 255).astype(np.uint8)
    
    def visualize_feature_statistics(
        self, 
        extracted_features: Dict[str, Any]
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create statistical visualization of extracted features
        
        Args:
            extracted_features: Dictionary of extracted features
            
        Returns:
            Tuple of (figure, axes)
        """
        num_features = len(extracted_features)
        if num_features == 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, "No features to visualize", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig, ax
            
        # Create subplots
        cols = min(3, num_features)
        rows = (num_features + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        if num_features == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
            
        plot_idx = 0
        
        # BEV semantic map statistics
        if "bev_semantic_map" in extracted_features:
            ax = axes[plot_idx]
            semantic_data = extracted_features["bev_semantic_map"]
            predictions = semantic_data["predictions"]
            
            # Class distribution
            unique_classes, counts = np.unique(predictions, return_counts=True)
            class_names = [self.bev_semantic_classes.get(c, {}).get("name", f"Class {c}") 
                          for c in unique_classes]
            
            bars = ax.bar(range(len(unique_classes)), counts)
            ax.set_xlabel("Semantic Classes")
            ax.set_ylabel("Pixel Count")
            ax.set_title("BEV Semantic Class Distribution")
            ax.set_xticks(range(len(unique_classes)))
            ax.set_xticklabels(class_names, rotation=45)
            
            # Color bars according to semantic colors
            for i, class_id in enumerate(unique_classes):
                if class_id in self.bev_semantic_classes:
                    bars[i].set_color(self.bev_semantic_classes[class_id]["color"])
                    
            plot_idx += 1
            
        # Agent states statistics
        if "agent_states" in extracted_features and plot_idx < len(axes):
            ax = axes[plot_idx]
            agent_states = extracted_features["agent_states"]
            
            if len(agent_states.shape) >= 2 and agent_states.shape[0] > 0:
                # Show distribution of agent positions
                ax.hist2d(agent_states[:, 0], agent_states[:, 1], bins=20, alpha=0.7)
                ax.set_xlabel("X Position")
                ax.set_ylabel("Y Position") 
                ax.set_title("Agent Position Distribution")
            else:
                ax.text(0.5, 0.5, "No agents detected", 
                       ha='center', va='center', transform=ax.transAxes)
            plot_idx += 1
            
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        return fig, axes
    
    def create_comprehensive_feature_view(
        self,
        extracted_features: Dict[str, Any],
        background_bev: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a comprehensive view of all extracted features
        
        Args:
            extracted_features: Dictionary of extracted features
            background_bev: Optional background BEV image
            save_path: Optional path to save the figure
            
        Returns:
            Tuple of (figure, axes_array)
        """
        # Determine layout based on available features
        has_semantic = "bev_semantic_map" in extracted_features
        has_agents = "agent_states" in extracted_features
        
        if has_semantic:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # BEV Semantic Map
            semantic_data = extracted_features["bev_semantic_map"]
            semantic_map = semantic_data["predictions"]
            confidence_map = semantic_data.get("confidence")
            
            # Main semantic visualization
            im1 = axes[0].imshow(
                semantic_map,
                cmap=self.semantic_colormap,
                vmin=0,
                vmax=len(self.bev_semantic_classes)-1,
                origin='lower'
            )
            axes[0].set_title("BEV Semantic Segmentation")
            axes[0].set_xlabel("BEV X (sideways)")
            axes[0].set_ylabel("BEV Y (forward)")
            self._add_semantic_legend(axes[0], semantic_map)
            
            # Confidence map
            if confidence_map is not None:
                im2 = axes[1].imshow(confidence_map, cmap='viridis', origin='lower')
                axes[1].set_title("Prediction Confidence")
                axes[1].set_xlabel("BEV X (sideways)")
                axes[1].set_ylabel("BEV Y (forward)")
                plt.colorbar(im2, ax=axes[1], label="Confidence")
            else:
                axes[1].text(0.5, 0.5, "No confidence data", 
                           ha='center', va='center', transform=axes[1].transAxes)
                
            # Overlay visualization
            if background_bev is not None:
                overlay = self.create_semantic_overlay(background_bev, semantic_map)
                axes[2].imshow(overlay, origin='lower')
                axes[2].set_title("Semantic Overlay on BEV")
                axes[2].set_xlabel("BEV X (sideways)")
                axes[2].set_ylabel("BEV Y (forward)")
            else:
                axes[2].text(0.5, 0.5, "No background BEV", 
                           ha='center', va='center', transform=axes[2].transAxes)
                
            # Statistics
            self._plot_feature_statistics_in_ax(axes[3], extracted_features)
            
        else:
            # Fallback for when no semantic map is available
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, "No BEV semantic features available", 
                   ha='center', va='center', transform=ax.transAxes)
            axes = [ax]
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comprehensive feature view to {save_path}")
            
        return fig, axes
    
    def _plot_feature_statistics_in_ax(self, ax: plt.Axes, extracted_features: Dict[str, Any]) -> None:
        """
        Plot feature statistics in a given axes
        
        Args:
            ax: Matplotlib axes
            extracted_features: Dictionary of extracted features
        """
        if "bev_semantic_map" in extracted_features:
            semantic_data = extracted_features["bev_semantic_map"]
            predictions = semantic_data["predictions"]
            
            # Class distribution
            unique_classes, counts = np.unique(predictions, return_counts=True)
            class_names = [self.bev_semantic_classes.get(c, {}).get("name", f"Class {c}") 
                          for c in unique_classes]
            
            bars = ax.bar(range(len(unique_classes)), counts)
            ax.set_xlabel("Semantic Classes")
            ax.set_ylabel("Pixel Count")
            ax.set_title("Class Distribution")
            ax.set_xticks(range(len(unique_classes)))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            
            # Color bars according to semantic colors
            for i, class_id in enumerate(unique_classes):
                if class_id in self.bev_semantic_classes:
                    bars[i].set_color(self.bev_semantic_classes[class_id]["color"])
        else:
            ax.text(0.5, 0.5, "No semantic statistics available", 
                   ha='center', va='center', transform=ax.transAxes)


def create_feature_visualization_config() -> Dict[str, Any]:
    """
    Create default configuration for feature visualization
    
    Returns:
        Configuration dictionary
    """
    return {
        "bev_semantic": {
            "overlay_alpha": 0.7,
            "show_legend": True,
            "show_confidence": True
        },
        "visualization": {
            "dpi": 300,
            "figure_size": (12, 8),
            "color_scheme": "viridis"
        },
        "output": {
            "save_format": "png",
            "save_quality": 95
        }
    }