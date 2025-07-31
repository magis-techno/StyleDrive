"""
Style Trajectory Data Manager

Simplified data manager for style-aware trajectory prediction.
Handles scene data loading with random scene selection capability.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, Scene, AgentInput

logger = logging.getLogger(__name__)


class StyleTrajectoryDataManager:
    """
    Simplified data manager for style-aware trajectory prediction
    """
    
    def __init__(self, dataset_path: str, inference_engine):
        """
        Initialize data manager
        
        Args:
            dataset_path: Path to NavSim dataset
            inference_engine: StyleTrajectoryInferenceEngine instance
        """
        self.dataset_path = Path(dataset_path)
        self.inference_engine = inference_engine
        
        # Initialize scene loader using agent's sensor config
        self.scene_loader = self._create_scene_loader()
        
        logger.info(f"Data manager initialized with {len(self.scene_loader.tokens)} scenes")
        logger.info(f"Dataset path: {self.dataset_path}")
        
    def _create_scene_loader(self):
        """
        Create SceneLoader using agent's sensor configuration
        """
        # Create empty scene filter (all scenes available)
        scene_filter = SceneFilter(
            log_names=None,
            tokens=None,
        )
        
        # Get sensor config from inference engine
        sensor_config = self.inference_engine.get_sensor_config()
        
        # Create scene loader
        scene_loader = SceneLoader(
            data_path=self.dataset_path,
            sensor_blobs_path=self.dataset_path,  # Assume same path structure
            scene_filter=scene_filter,
            sensor_config=sensor_config
        )
        
        return scene_loader
    
    def get_available_scenes(self, limit: Optional[int] = None) -> List[str]:
        """
        Get list of available scene tokens
        
        Args:
            limit: Optional limit on number of scenes to return
            
        Returns:
            List of scene tokens
        """
        tokens = self.scene_loader.tokens
        if limit:
            tokens = tokens[:limit]
        return tokens
    
    def get_random_scene(self) -> Tuple[Scene, str]:
        """
        Randomly select and load a scene
        
        Returns:
            Tuple of (Scene object, scene_token)
        """
        # Get random token
        tokens = self.scene_loader.tokens
        if not tokens:
            raise ValueError("No scenes available in dataset")
            
        random_token = np.random.choice(tokens)
        logger.info(f"Randomly selected scene: {random_token}")
        
        # Load scene
        scene = self.load_scene(random_token)
        return scene, random_token
    
    def load_scene(self, scene_token: str) -> Scene:
        """
        Load scene object for a given token
        
        Args:
            scene_token: Scene token to load
            
        Returns:
            Scene object
        """
        try:
            scene = self.scene_loader.get_scene_from_token(scene_token)
            logger.debug(f"Successfully loaded scene: {scene_token}")
            return scene
            
        except Exception as e:
            logger.error(f"Failed to load scene {scene_token}: {e}")
            raise
    
    def load_agent_input(self, scene_token: str) -> AgentInput:
        """
        Load AgentInput for a given scene token
        
        Args:
            scene_token: Scene token to load
            
        Returns:
            AgentInput object
        """
        scene = self.load_scene(scene_token)
        return scene.get_agent_input()
    
    def get_scene_info(self, scene_token: str) -> Dict[str, Any]:
        """
        Get basic information about a scene
        
        Args:
            scene_token: Scene token to inspect
            
        Returns:
            Dictionary with scene information
        """
        try:
            scene = self.load_scene(scene_token)
            metadata = scene.scene_metadata
            agent_input = scene.get_agent_input()
            
            return {
                "token": scene_token,
                "log_name": metadata.log_name,
                "map_name": metadata.map_name,
                "initial_token": metadata.initial_token,
                "camera_frames": len(agent_input.cameras),
                "lidar_frames": len(agent_input.lidars),
                "ego_status_frames": len(agent_input.ego_statuses)
            }
            
        except Exception as e:
            logger.error(f"Failed to get scene info for {scene_token}: {e}")
            return {"token": scene_token, "error": str(e)}
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        tokens = self.scene_loader.tokens
        
        # Sample a few scenes to get statistics
        sample_size = min(10, len(tokens))
        sample_tokens = np.random.choice(tokens, sample_size, replace=False)
        
        map_names = []
        log_names = []
        
        for token in sample_tokens:
            try:
                scene_info = self.get_scene_info(token)
                if "map_name" in scene_info:
                    map_names.append(scene_info["map_name"])
                if "log_name" in scene_info:
                    log_names.append(scene_info["log_name"])
            except Exception as e:
                logger.debug(f"Failed to load scene {token} for statistics: {e}")
        
        unique_maps = list(set(map_names))
        unique_logs = list(set(log_names))
        
        return {
            "total_scenes": len(tokens),
            "sample_size": len(sample_tokens),
            "map_locations": unique_maps,
            "num_map_locations": len(unique_maps),
            "log_names": unique_logs,
            "num_logs": len(unique_logs),
            "dataset_path": str(self.dataset_path)
        }
    
    def validate_scene(self, scene_token: str) -> bool:
        """
        Validate that a scene can be properly loaded
        
        Args:
            scene_token: Scene token to validate
            
        Returns:
            True if scene is valid, False otherwise
        """
        try:
            scene = self.load_scene(scene_token)
            agent_input = scene.get_agent_input()
            
            # Basic validation checks
            if not agent_input.cameras:
                logger.warning(f"Scene {scene_token} has no camera data")
                return False
                
            if not agent_input.lidars:
                logger.warning(f"Scene {scene_token} has no lidar data")
                return False
                
            if not agent_input.ego_statuses:
                logger.warning(f"Scene {scene_token} has no ego status data")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Scene {scene_token} validation failed: {e}")
            return False