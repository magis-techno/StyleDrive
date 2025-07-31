"""
Style Trajectory Inference Engine

Inference engine for style-aware trajectory prediction models.
Supports DiffusionDrive-Style with three driving styles: Aggressive, Normal, Conservative.
"""

import logging
import time
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, Scene

logger = logging.getLogger(__name__)


class StyleTrajectoryInferenceEngine:
    """
    Style-aware trajectory prediction inference engine
    Handles model loading, initialization, and multi-style inference
    """
    
    # Style mapping consistent with training
    STYLE_MAP = {
        'aggressive': {'code': 'A', 'index': 0, 'color': 'red'},
        'normal': {'code': 'N', 'index': 1, 'color': 'blue'},
        'conservative': {'code': 'C', 'index': 2, 'color': 'green'}
    }
    
    def __init__(self, checkpoint_path: str, lr: float = 6e-4):
        """
        Initialize the inference engine
        
        Args:
            checkpoint_path: Path to model checkpoint
            lr: Learning rate (used during agent initialization)
        """
        self.model_type = "diffusiondrive_style"
        self.checkpoint_path = checkpoint_path
        self.lr = lr
        self.agent = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing style-aware inference engine")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """
        Load and initialize the DiffusionDrive-Style model
        """
        start_time = time.time()
        
        self._load_diffusiondrive_style_model()
        
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
        
    def _load_diffusiondrive_style_model(self):
        """Load DiffusionDrive model with style support enabled"""
        try:
            from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
            from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
            
            # Create configuration with style support
            config = TransfuserConfig()
            config.with_style = True  # 🎯 Enable style support
            config.styletrain_path = ""  # Not needed for inference
            config.styletest_path = ""   # Not needed for inference
            
            # Create agent instance
            self.agent = TransfuserAgent(
                config=config,
                lr=self.lr,
                checkpoint_path=self.checkpoint_path
            )
            
            # Initialize weights from checkpoint
            if self.checkpoint_path:
                logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
                self.agent.initialize()
            else:
                logger.warning("No checkpoint provided, using randomly initialized weights")
                
        except Exception as e:
            logger.error(f"Failed to load DiffusionDrive-Style model: {e}")
            raise
    
    def _build_style_feature(self, style_name: str) -> torch.Tensor:
        """
        Build one-hot style feature vector
        
        Args:
            style_name: One of 'aggressive', 'normal', 'conservative'
            
        Returns:
            One-hot encoded style vector [1, 3]
        """
        if style_name not in self.STYLE_MAP:
            raise ValueError(f"Unknown style: {style_name}. Must be one of {list(self.STYLE_MAP.keys())}")
            
        style_idx = self.STYLE_MAP[style_name]['index']
        style_feature = F.one_hot(torch.tensor(style_idx), num_classes=3).float()
        return style_feature.unsqueeze(0)  # Add batch dimension
    
    def predict(self, agent_input: AgentInput, style_name: str = 'normal') -> Dict[str, Any]:
        """
        Predict trajectory for a specific driving style
        
        Args:
            agent_input: Input data for the scene
            style_name: Driving style ('aggressive', 'normal', 'conservative')
            
        Returns:
            Dictionary containing prediction results
        """
        if self.agent is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        logger.debug(f"Starting style prediction: {style_name}")
        
        try:
            # Set agent to eval mode
            self.agent.eval()
            
            # Build features using agent's feature builders
            features = {}
            for builder in self.agent.get_feature_builders():
                features.update(builder.compute_features(agent_input))
            
            # Add style feature
            style_feature = self._build_style_feature(style_name)
            features["style_feature"] = style_feature
            
            logger.debug(f"Built {len(features)} feature tensors (including style)")
            
            # Add batch dimension if needed
            features = {k: v.unsqueeze(0) if v.dim() == len(v.shape) and k != "style_feature" else v 
                       for k, v in features.items()}
            
            # Log original device info
            original_devices = {k: v.device for k, v in features.items()}
            logger.debug(f"Original feature devices: {original_devices}")
            
            # Move features to same device as model
            features = {k: v.to(self.device) for k, v in features.items()}
            logger.debug(f"Moved features to device: {self.device}")
            
            # Perform inference
            with torch.no_grad():
                predictions = self.agent.forward(features)
                
                # Extract trajectory and move back to CPU
                trajectory_tensor = predictions["trajectory"].squeeze(0).cpu()
                poses = trajectory_tensor.numpy()
                
                # Extract additional features for visualization
                extracted_features = {}
                
                # Extract BEV semantic map if available
                if "bev_semantic_map" in predictions:
                    bev_semantic_logits = predictions["bev_semantic_map"].squeeze(0).cpu()
                    bev_semantic_map = torch.argmax(bev_semantic_logits, dim=0).numpy()
                    extracted_features["bev_semantic_map"] = {
                        "predictions": bev_semantic_map,
                        "logits": bev_semantic_logits.numpy(),
                        "confidence": torch.softmax(bev_semantic_logits, dim=0).max(dim=0)[0].numpy()
                    }
                    logger.debug(f"Extracted BEV semantic map: {bev_semantic_map.shape}")
                
                # Extract agent predictions if available
                if "agent_states" in predictions:
                    agent_states = predictions["agent_states"].squeeze(0).cpu().numpy()
                    extracted_features["agent_states"] = agent_states
                
                if "agent_labels" in predictions:
                    agent_labels = predictions["agent_labels"].squeeze(0).cpu().numpy()
                    extracted_features["agent_labels"] = agent_labels
            
            # Build trajectory object
            from navsim.common.dataclasses import Trajectory
            pred_trajectory = Trajectory(poses)
            
            inference_time = time.time() - start_time
            
            # Collect results
            result = {
                "trajectory": pred_trajectory,
                "style": style_name,
                "inference_time": inference_time,
                "model_type": self.model_type,
                "trajectory_length": len(pred_trajectory.poses),
                "extracted_features": extracted_features,
                "device_info": {
                    "model_device": str(self.device),
                    "inference_device": str(self.device)
                }
            }
            
            logger.debug(f"Style '{style_name}' inference completed in {inference_time:.3f}s")
            logger.debug(f"Predicted trajectory with {len(pred_trajectory.poses)} points")
            
            return result
            
        except Exception as e:
            logger.error(f"Style '{style_name}' inference failed: {e}")
            logger.error(f"Model device: {self.device}")
            if 'features' in locals():
                logger.error(f"Feature devices: {[f'{k}: {v.device}' for k, v in features.items()]}")
            raise
    
    def predict_all_styles(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        Predict trajectories for all three driving styles
        
        Args:
            agent_input: Input data for the scene
            
        Returns:
            Dictionary mapping style names to trajectory tensors
        """
        logger.info("Predicting trajectories for all driving styles...")
        
        results = {}
        total_start_time = time.time()
        
        for style_name in self.STYLE_MAP.keys():
            style_result = self.predict(agent_input, style_name)
            
            # Extract trajectory tensor for visualization compatibility
            trajectory = style_result["trajectory"]
            trajectory_tensor = torch.tensor(trajectory.poses).unsqueeze(0)  # Add batch dimension
            results[style_name] = trajectory_tensor
            
            logger.info(f"✅ {style_name.capitalize()} style completed")
        
        total_time = time.time() - total_start_time
        logger.info(f"All styles prediction completed in {total_time:.2f}s")
        
        return results
    
    def get_sensor_config(self):
        """Get sensor configuration required for data loading"""
        if self.agent is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.agent.get_sensor_config()
    
    def get_feature_builders(self):
        """Get feature builders from the agent"""
        if self.agent is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.agent.get_feature_builders()
    
    def get_available_styles(self) -> list:
        """Get list of available driving styles"""
        return list(self.STYLE_MAP.keys())
    
    def get_style_info(self, style_name: str) -> Dict[str, Any]:
        """Get information about a specific style"""
        return self.STYLE_MAP.get(style_name, {})
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.agent is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_type": self.model_type,
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "sensor_config": self.agent.get_sensor_config(),
            "num_parameters": sum(p.numel() for p in self.agent.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.agent.parameters() if p.requires_grad),
            "style_support": True,
            "available_styles": list(self.STYLE_MAP.keys())
        }