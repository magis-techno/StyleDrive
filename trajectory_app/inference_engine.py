"""
Trajectory Inference Engine

This module provides the core inference engine for trajectory prediction models.
"""

import logging
import time
from typing import Dict, Any, Optional
import torch

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, Scene

logger = logging.getLogger(__name__)


class TrajectoryInferenceEngine:
    """
    Trajectory prediction inference engine
    Handles model loading, initialization, and inference
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the inference engine
        
        Args:
            model_config: Configuration dictionary containing model parameters
                - type: Model type ("diffusiondrive", "transfuser", etc.)
                - checkpoint_path: Path to model checkpoint
                - lr: Learning rate (used during agent initialization)
        """
        self.model_type = model_config.get("type", "diffusiondrive")
        self.checkpoint_path = model_config.get("checkpoint_path")
        self.lr = model_config.get("lr", 6e-4)
        self.agent = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing inference engine for {self.model_type} model")
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """
        Load and initialize the model following the evaluation script pattern
        """
        start_time = time.time()
        
        if self.model_type == "diffusiondrive":
            self._load_diffusiondrive_model()
        elif self.model_type == "transfuser":
            self._load_transfuser_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Set to evaluation mode and move to device
        self.agent.eval()
        self.agent.to(self.device)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f}s")
        logger.info(f"Model device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Sensor config: {self.agent.get_sensor_config()}")
        
        # Device verification
        model_params_device = next(self.agent.parameters()).device
        logger.info(f"Model parameters are on: {model_params_device}")
        
        if model_params_device != self.device:
            logger.warning(f"Device mismatch detected! Expected: {self.device}, Found: {model_params_device}")
        else:
            logger.info(f"✅ Model and expected device match: {self.device}")
        
    def _load_diffusiondrive_model(self):
        """Load DiffusionDrive model"""
        try:
            from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
            from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
            
            # Create configuration
            config = TransfuserConfig()
            
            # Create agent instance
            self.agent = TransfuserAgent(
                config=config,
                lr=self.lr,
                checkpoint_path=self.checkpoint_path
            )
            
            # Initialize weights if checkpoint provided
            if self.checkpoint_path:
                logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
                self.agent.initialize()
            else:
                logger.warning("No checkpoint provided, using randomly initialized weights")
                
        except Exception as e:
            logger.error(f"Failed to load DiffusionDrive model: {e}")
            raise
            
    def _load_transfuser_model(self):
        """Load Transfuser model"""
        try:
            from navsim.agents.transfuser.transfuser_config import TransfuserConfig
            from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
            
            # Create configuration
            config = TransfuserConfig()
            
            # Create agent instance
            self.agent = TransfuserAgent(
                config=config,
                lr=self.lr,
                checkpoint_path=self.checkpoint_path
            )
            
            # Initialize weights if checkpoint provided
            if self.checkpoint_path:
                logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
                self.agent.initialize()
            else:
                logger.warning("No checkpoint provided, using randomly initialized weights")
                
        except Exception as e:
            logger.error(f"Failed to load Transfuser model: {e}")
            raise
    
    def predict_trajectory(self, agent_input: AgentInput, scene: Optional[Scene] = None) -> Dict[str, Any]:
        """
        Predict trajectory for given agent input
        
        Args:
            agent_input: Input data for the agent
            scene: Optional scene data (required for some models)
            
        Returns:
            Dictionary containing prediction results and metadata
        """
        if self.agent is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # SOLUTION 1: Handle device mismatch by manually building features and transferring to device
        # This avoids modifying core NavSim code while ensuring model and data are on same device
        logger.debug(f"Model device: {self.device}")
        
        try:
            # Set agent to eval mode
            self.agent.eval()
            
            # Manually build features (same as AbstractAgent.compute_trajectory)
            features = {}
            for builder in self.agent.get_feature_builders():
                features.update(builder.compute_features(agent_input))
            
            logger.debug(f"Built {len(features)} feature tensors")
            
            # Add batch dimension
            features = {k: v.unsqueeze(0) for k, v in features.items()}
            
            # Log original device info
            original_devices = {k: v.device for k, v in features.items()}
            logger.debug(f"Original feature devices: {original_devices}")
            
            # CRITICAL FIX: Move features to same device as model
            features = {k: v.to(self.device) for k, v in features.items()}
            logger.debug(f"Moved features to device: {self.device}")
            
            # Perform inference with device-matched tensors
            with torch.no_grad():
                predictions = self.agent.forward(features)
                
                # Extract trajectory and move back to CPU for numpy conversion
                trajectory_tensor = predictions["trajectory"].squeeze(0).cpu()
                poses = trajectory_tensor.numpy()
                
                # Extract additional features for visualization
                extracted_features = {}
                
                # Extract BEV semantic map if available
                if "bev_semantic_map" in predictions:
                    bev_semantic_logits = predictions["bev_semantic_map"].squeeze(0).cpu()
                    # Convert logits to class predictions using argmax
                    bev_semantic_map = torch.argmax(bev_semantic_logits, dim=0).numpy()
                    extracted_features["bev_semantic_map"] = {
                        "predictions": bev_semantic_map,  # [H, W] class indices
                        "logits": bev_semantic_logits.numpy(),  # [num_classes, H, W] raw logits
                        "confidence": torch.softmax(bev_semantic_logits, dim=0).max(dim=0)[0].numpy()  # [H, W] confidence
                    }
                    logger.debug(f"Extracted BEV semantic map: {bev_semantic_map.shape}, classes: {torch.unique(torch.from_numpy(bev_semantic_map)).numpy()}")
                
                # Extract agent predictions if available
                if "agent_states" in predictions:
                    agent_states = predictions["agent_states"].squeeze(0).cpu().numpy()
                    extracted_features["agent_states"] = agent_states
                
                if "agent_labels" in predictions:
                    agent_labels = predictions["agent_labels"].squeeze(0).cpu().numpy()
                    extracted_features["agent_labels"] = agent_labels
            
            # Build trajectory object (same as AbstractAgent.compute_trajectory)
            from navsim.common.dataclasses import Trajectory
            pred_trajectory = Trajectory(poses)
            
            inference_time = time.time() - start_time
            
            # Collect results
            result = {
                "trajectory": pred_trajectory,
                "inference_time": inference_time,
                "model_type": self.model_type,
                "trajectory_length": len(pred_trajectory.poses),
                "time_horizon": pred_trajectory.trajectory_sampling.time_horizon if hasattr(pred_trajectory, 'trajectory_sampling') else None,
                "extracted_features": extracted_features,  # New: add extracted features
                "device_info": {
                    "model_device": str(self.device),
                    "original_feature_devices": {k: str(v) for k, v in original_devices.items()},
                    "inference_device": str(self.device)
                }
            }
            
            logger.debug(f"Inference completed in {inference_time:.3f}s on {self.device}")
            logger.debug(f"Predicted trajectory with {len(pred_trajectory.poses)} points")
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            logger.error(f"Model device: {self.device}")
            if 'features' in locals():
                logger.error(f"Feature devices: {[f'{k}: {v.device}' for k, v in features.items()]}")
            raise
    
    def get_sensor_config(self):
        """
        Get sensor configuration required for data loading
        """
        if self.agent is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.agent.get_sensor_config()
    
    def get_feature_builders(self):
        """Get feature builders from the agent"""
        if self.agent is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.agent.get_feature_builders()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        """
        if self.agent is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_type": self.model_type,
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "sensor_config": self.agent.get_sensor_config(),
            "num_parameters": sum(p.numel() for p in self.agent.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.agent.parameters() if p.requires_grad)
        } 