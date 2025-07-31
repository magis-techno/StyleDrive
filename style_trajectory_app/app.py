"""
Style Trajectory Prediction Application

Simplified application class for style-aware trajectory prediction.
Coordinates model inference, data management, and visualization.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .inference_engine import StyleTrajectoryInferenceEngine
from .data_manager import StyleTrajectoryDataManager
from .style_visualization import plot_style_trajectories

logger = logging.getLogger(__name__)


class StyleTrajectoryApp:
    """
    Simplified trajectory prediction application for style-aware inference
    
    Coordinates model loading, data management, and style comparison visualization
    """
    
    def __init__(
        self, 
        checkpoint_path: str, 
        dataset_path: str,
        lr: float = 6e-4
    ):
        """
        Initialize the application
        
        Args:
            checkpoint_path: Path to DiffusionDrive-Style model checkpoint
            dataset_path: Path to NavSim dataset
            lr: Learning rate (used during agent initialization)
        """
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.lr = lr
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize components
        self.inference_engine = None
        self.data_manager = None
        
        # Initialize all components
        self._initialize_components()
        
        logger.info("StyleTrajectoryApp initialized successfully!")
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def _initialize_components(self):
        """Initialize all application components in correct order"""
        logger.info("Initializing style trajectory application components...")
        
        # 1. Initialize inference engine and load model
        logger.info("Loading DiffusionDrive-Style model...")
        self.inference_engine = StyleTrajectoryInferenceEngine(
            checkpoint_path=self.checkpoint_path,
            lr=self.lr
        )
        self.inference_engine.load_model()
        
        # 2. Initialize data manager with inference engine
        logger.info("Initializing data manager...")
        self.data_manager = StyleTrajectoryDataManager(
            dataset_path=self.dataset_path,
            inference_engine=self.inference_engine
        )
        
        # Log initialization summary
        stats = self.data_manager.get_dataset_statistics()
        model_info = self.inference_engine.get_model_info()
        
        logger.info(f"Initialization complete!")
        logger.info(f"Model: {model_info['model_type']} ({model_info['status']})")
        logger.info(f"Available scenes: {stats['total_scenes']}")
        logger.info(f"Available styles: {model_info['available_styles']}")
    
    def get_random_scene(self) -> Tuple[Any, str]:
        """
        Get a random scene from the dataset
        
        Returns:
            Tuple of (Scene object, scene_token)
        """
        return self.data_manager.get_random_scene()
    
    def predict_single_style(
        self, 
        scene_token: str, 
        style_name: str = 'normal'
    ) -> Dict[str, Any]:
        """
        Predict trajectory for a single scene with specific style
        
        Args:
            scene_token: Scene token to process
            style_name: Driving style ('aggressive', 'normal', 'conservative')
            
        Returns:
            Dictionary containing prediction results
        """
        logger.info(f"Predicting {style_name} style for scene: {scene_token}")
        
        # Load scene data
        agent_input = self.data_manager.load_agent_input(scene_token)
        
        # Predict with specific style
        result = self.inference_engine.predict(agent_input, style_name)
        
        # Add scene metadata
        scene_info = self.data_manager.get_scene_info(scene_token)
        result["scene_metadata"] = scene_info
        
        return result
    
    def predict_all_styles(self, scene_token: str) -> Dict[str, Any]:
        """
        Predict trajectories for all three driving styles
        
        Args:
            scene_token: Scene token to process
            
        Returns:
            Dictionary containing results for all styles
        """
        logger.info(f"Predicting all styles for scene: {scene_token}")
        start_time = time.time()
        
        # Load scene data
        agent_input = self.data_manager.load_agent_input(scene_token)
        
        # Predict all styles
        trajectories = self.inference_engine.predict_all_styles(agent_input)
        
        # Get scene metadata
        scene_info = self.data_manager.get_scene_info(scene_token)
        
        total_time = time.time() - start_time
        
        result = {
            "trajectories": trajectories,
            "scene_metadata": scene_info,
            "total_prediction_time": total_time,
            "styles": list(trajectories.keys())
        }
        
        logger.info(f"All styles prediction completed in {total_time:.2f}s")
        return result
    
    def visualize_style_comparison(
        self, 
        trajectories: Dict[str, Any], 
        scene_data=None,
        title: str = "Driving Style Trajectory Comparison",
        save_path: Optional[str] = None
    ):
        """
        Create visualization comparing different style trajectories
        
        Args:
            trajectories: Dictionary mapping style names to trajectory tensors
            scene_data: Optional scene data for background
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating style comparison visualization...")
        
        fig = plot_style_trajectories(
            trajectories=trajectories,
            scene_data=scene_data,
            title=title,
            save_path=save_path
        )
        
        if save_path:
            logger.info(f"Visualization saved to: {save_path}")
        
        return fig
    
    def run_style_demo(self, scene_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete style demonstration: random scene + all styles + visualization
        
        Args:
            scene_token: Optional specific scene token. If None, random scene is selected
            
        Returns:
            Dictionary containing complete demo results
        """
        logger.info("Running style trajectory demonstration...")
        demo_start_time = time.time()
        
        # 1. Get scene (random or specified)
        if scene_token is None:
            scene, scene_token = self.get_random_scene()
            logger.info(f"Random scene selected: {scene_token}")
        else:
            scene = self.data_manager.load_scene(scene_token)
            logger.info(f"Using specified scene: {scene_token}")
        
        # 2. Predict all styles
        all_styles_result = self.predict_all_styles(scene_token)
        
        # 3. Create visualization
        fig = self.visualize_style_comparison(
            trajectories=all_styles_result["trajectories"],
            scene_data=scene,
            title=f"Style Comparison - Scene {scene_token[:8]}..."
        )
        
        demo_time = time.time() - demo_start_time
        
        result = {
            "scene_token": scene_token,
            "scene_metadata": all_styles_result["scene_metadata"],
            "trajectories": all_styles_result["trajectories"],
            "visualization": fig,
            "demo_time": demo_time,
            "prediction_time": all_styles_result["total_prediction_time"]
        }
        
        logger.info(f"Style demo completed in {demo_time:.2f}s")
        logger.info(f"Scene: {scene_token}")
        logger.info(f"Map: {result['scene_metadata'].get('map_name', 'unknown')}")
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return self.inference_engine.get_model_info()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        return self.data_manager.get_dataset_statistics()
    
    def get_available_styles(self) -> List[str]:
        """Get list of available driving styles"""
        return self.inference_engine.get_available_styles()
    
    def get_style_info(self, style_name: str) -> Dict[str, Any]:
        """Get information about a specific style"""
        return self.inference_engine.get_style_info(style_name)
    
    def validate_scene(self, scene_token: str) -> bool:
        """Validate that a scene can be properly loaded and processed"""
        return self.data_manager.validate_scene(scene_token)
    
    def get_random_scenes(self, count: int = 5) -> List[str]:
        """
        Get multiple random scene tokens
        
        Args:
            count: Number of random scenes to return
            
        Returns:
            List of scene tokens
        """
        available_scenes = self.data_manager.get_available_scenes()
        if count >= len(available_scenes):
            return available_scenes
        
        import numpy as np
        return list(np.random.choice(available_scenes, count, replace=False))
    
    def __str__(self) -> str:
        """String representation of the application"""
        model_info = self.get_model_info()
        dataset_info = self.get_dataset_info()
        
        return (
            f"StyleTrajectoryApp(\n"
            f"  Model: {model_info.get('model_type', 'unknown')}\n"
            f"  Status: {model_info.get('status', 'unknown')}\n"
            f"  Styles: {model_info.get('available_styles', [])}\n"
            f"  Dataset: {dataset_info.get('total_scenes', 0)} scenes\n"
            f"  Device: {model_info.get('device', 'unknown')}\n"
            f")"
        )