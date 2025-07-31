"""
Data Manager

This module handles scene data loading, trajectory management, and PDM cache interaction.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, Trajectory
from navsim.common.dataloader import MetricCacheLoader

logger = logging.getLogger(__name__)


class TrajectoryDataManager:
    """
    Manages scene data loading and trajectory synchronization
    """
    
    def __init__(self, data_config: Dict[str, Any], inference_engine):
        """
        Initialize data manager
        
        Args:
            data_config: Configuration dictionary containing:
                - navsim_log_path: Path to NavSim log data
                - sensor_blobs_path: Path to sensor blob data  
                - cache_path: Path to metric cache
            inference_engine: TrajectoryInferenceEngine instance
        """
        self.data_config = data_config
        self.inference_engine = inference_engine
        
        # Initialize scene loader using agent's sensor config
        self.scene_loader = self._create_scene_loader()
        
        # Initialize metric cache loader
        cache_path = data_config.get("cache_path")
        if cache_path and Path(cache_path).exists():
            self.metric_cache_loader = MetricCacheLoader(Path(cache_path))
            logger.info(f"Loaded metric cache from: {cache_path}")
        else:
            self.metric_cache_loader = None
            logger.warning(f"Metric cache not found at: {cache_path}")
            
        logger.info(f"Data manager initialized with {len(self.scene_loader.tokens)} scenes")
        
    def _create_scene_loader(self):
        """
        Create SceneLoader using agent's sensor configuration
        """
        # Create scene filter (initially empty, will be set per request)
        scene_filter = SceneFilter(
            log_names=None,
            tokens=None,
        )
        
        # Get sensor config from inference engine
        sensor_config = self.inference_engine.get_sensor_config()
        
        # Create scene loader
        scene_loader = SceneLoader(
            data_path=Path(self.data_config["navsim_log_path"]),
            sensor_blobs_path=Path(self.data_config["sensor_blobs_path"]),
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
    
    def load_scene_data(self, scene_token: str) -> Dict[str, Any]:
        """
        Load complete scene data for a given token
        
        Args:
            scene_token: Scene token to load
            
        Returns:
            Dictionary containing scene data and metadata
        """
        # Verify token exists
        if scene_token not in self.scene_loader.tokens:
            available_tokens = self.scene_loader.tokens[:5]
            raise ValueError(
                f"Scene token {scene_token} not found. "
                f"Available tokens (first 5): {available_tokens}"
            )
        
        logger.debug(f"Loading scene data for token: {scene_token}")
        
        # Load scene
        scene = self.scene_loader.get_scene_from_token(scene_token)
        
        # Get current frame (last history frame)
        current_frame_idx = scene.scene_metadata.num_history_frames - 1
        current_frame = scene.frames[current_frame_idx]
        
        # Extract sensor data
        sensors = {
            "cameras": current_frame.cameras,
            "lidar": current_frame.lidar
        }
        
        # Extract map and ego information
        map_info = {
            "api": scene.map_api,
            "ego_pose": current_frame.ego_status.ego_pose,
            "ego_status": current_frame.ego_status
        }
        
        # Extract metadata - FIXED: NavSim data structure corrections
        # - SceneMetadata has no scenario_type attribute -> use "unknown"
        # - EgoStatus has no timestamp attribute -> use current_frame.timestamp
        scene_metadata = scene.scene_metadata
        metadata = {
            "token": scene_token,
            "scenario_type": "unknown",  # Fixed: NavSim always uses "unknown"
            "log_name": scene_metadata.log_name,
            "map_name": scene_metadata.map_name,
            "timestamp": current_frame.timestamp,  # Fixed: timestamp is in Frame, not EgoStatus
            "num_history_frames": scene_metadata.num_history_frames,
            "num_future_frames": scene_metadata.num_future_frames,
            "total_frames": len(scene.frames)
        }
        
        return {
            "scene": scene,
            "sensors": sensors,
            "map": map_info,
            "metadata": metadata
        }
    
    def get_all_trajectories(self, scene_token: str) -> Dict[str, Any]:
        """
        Get all available trajectories for a scene (GT, PDM-Closed, etc.)
        
        Args:
            scene_token: Scene token
            
        Returns:
            Dictionary containing all available trajectories
        """
        scene_data = self.load_scene_data(scene_token)
        scene = scene_data["scene"]
        
        trajectories = {}
        
        # Ground truth trajectory
        try:
            gt_trajectory = scene.get_future_trajectory()
            trajectories["ground_truth"] = gt_trajectory
            logger.debug(f"Loaded GT trajectory with {len(gt_trajectory.poses)} points")
        except Exception as e:
            logger.warning(f"Could not load ground truth trajectory: {e}")
        
        # PDM-Closed trajectory
        if self.metric_cache_loader:
            try:
                pdm_trajectory = self._load_pdm_trajectory(scene_token)
                if pdm_trajectory:
                    trajectories["pdm_closed"] = pdm_trajectory
                    logger.debug(f"Loaded PDM trajectory with {len(pdm_trajectory.poses)} points")
            except Exception as e:
                logger.warning(f"Could not load PDM trajectory: {e}")
        
        return trajectories
    
    def _load_pdm_trajectory(self, scene_token: str) -> Optional[Trajectory]:
        """
        Load PDM-Closed trajectory from metric cache
        
        Args:
            scene_token: Scene token
            
        Returns:
            PDM trajectory or None if not available
        """
        if not self.metric_cache_loader:
            return None
            
        try:
            # Check if token exists in cache
            if scene_token not in self.metric_cache_loader.tokens:
                logger.debug(f"Token {scene_token} not found in metric cache")
                return None
                
            # Load metric cache
            metric_cache = self.metric_cache_loader.get_from_token(scene_token)
            
            # Convert InterpolatedTrajectory to Trajectory
            interpolated_traj = metric_cache.trajectory
            poses = []
            
            for ego_state in interpolated_traj._ego_states:
                pose = [
                    ego_state.rear_axle.x,
                    ego_state.rear_axle.y,
                    ego_state.rear_axle.heading
                ]
                poses.append(pose)
            
            return Trajectory(poses=poses)
            
        except Exception as e:
            logger.debug(f"Failed to load PDM trajectory for {scene_token}: {e}")
            return None
    
    def synchronize_trajectories(
        self, 
        trajectories: Dict[str, Trajectory], 
        time_horizon: float = 6.0,
        dt: float = 0.1
    ) -> Dict[str, Dict[str, Any]]:
        """
        Synchronize trajectories to common time grid
        
        Args:
            trajectories: Dictionary of trajectory name -> Trajectory
            time_horizon: Maximum time horizon in seconds
            dt: Time step in seconds
            
        Returns:
            Dictionary with synchronized trajectory data
        """
        time_grid = np.arange(0, time_horizon + dt, dt)
        synchronized = {}
        
        for name, trajectory in trajectories.items():
            if not hasattr(trajectory, 'poses') or len(trajectory.poses) == 0:
                logger.warning(f"Empty or invalid trajectory for {name}")
                continue
                
            poses = np.array(trajectory.poses)
            
            # Get original time sampling
            if hasattr(trajectory, 'trajectory_sampling'):
                original_dt = trajectory.trajectory_sampling.interval_length
                original_times = np.arange(len(poses)) * original_dt
            else:
                # Assume default sampling if not available
                original_dt = 0.5  # Default NavSim sampling
                original_times = np.arange(len(poses)) * original_dt
            
            # Interpolate to common time grid
            sync_poses = self._interpolate_trajectory(poses, original_times, time_grid)
            
            synchronized[name] = {
                "poses": sync_poses,
                "timestamps": time_grid[:len(sync_poses)],
                "original_length": len(poses),
                "original_dt": original_dt,
                "interpolated": len(sync_poses) != len(poses)
            }
            
            logger.debug(f"Synchronized {name}: {len(poses)} -> {len(sync_poses)} points")
        
        return synchronized
    
    def _interpolate_trajectory(
        self, 
        poses: np.ndarray, 
        original_times: np.ndarray, 
        target_times: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate trajectory to target time grid
        
        Args:
            poses: Original trajectory poses [N, 3]
            original_times: Original time stamps [N]
            target_times: Target time grid [M]
            
        Returns:
            Interpolated poses [M, 3]
        """
        # Ensure we don't extrapolate beyond original data
        max_time = original_times[-1]
        target_times = target_times[target_times <= max_time]
        
        if len(target_times) == 0:
            return poses[:1]  # Return first pose if no valid target times
        
        # Interpolate each coordinate
        interpolated_poses = []
        for i in range(poses.shape[1]):  # x, y, heading
            if i == 2:  # Handle heading (circular interpolation)
                # Convert to complex representation for circular interpolation
                complex_heading = np.exp(1j * poses[:, i])
                interp_complex = np.interp(target_times, original_times, complex_heading.real) + \
                               1j * np.interp(target_times, original_times, complex_heading.imag)
                interp_heading = np.angle(interp_complex)
                interpolated_poses.append(interp_heading)
            else:  # x, y coordinates
                interp_coord = np.interp(target_times, original_times, poses[:, i])
                interpolated_poses.append(interp_coord)
        
        return np.column_stack(interpolated_poses)
    
    def get_scene_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about available scenes
        
        Returns:
            Dictionary with scene statistics
        """
        tokens = self.scene_loader.tokens
        
        # Sample a few scenes to get statistics
        sample_size = min(10, len(tokens))
        sample_tokens = np.random.choice(tokens, sample_size, replace=False)
        
        map_names = []
        log_names = []
        
        for token in sample_tokens:
            try:
                scene_data = self.load_scene_data(token)
                map_names.append(scene_data["metadata"]["map_name"])
                log_names.append(scene_data["metadata"]["log_name"])
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
            "has_metric_cache": self.metric_cache_loader is not None,
            "metric_cache_scenes": len(self.metric_cache_loader.tokens) if self.metric_cache_loader else 0
        } 