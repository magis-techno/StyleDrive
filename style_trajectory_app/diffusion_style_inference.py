"""
DiffusionDrive-Style Inference Module

This module provides a simplified interface for running inference with
the DiffusionDrive-Style model, enabling style-aware trajectory prediction.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from pathlib import Path

from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.common.dataclasses import AgentInput


class DiffusionStyleInference:
    """
    Simplified inference class for DiffusionDrive-Style model.
    
    This class handles loading the pre-trained model and running inference
    with different driving styles (Aggressive, Normal, Conservative).
    """
    
    # Style mapping consistent with training
    STYLE_MAP = {
        'aggressive': {'code': 'A', 'index': 0, 'color': 'red'},
        'normal': {'code': 'N', 'index': 1, 'color': 'blue'},
        'conservative': {'code': 'C', 'index': 2, 'color': 'green'}
    }
    
    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None):
        """
        Initialize the DiffusionDrive-Style inference engine.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            config_path: Optional path to model config (will use defaults if None)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.agent = self._load_model()
        
    def _load_model(self) -> TransfuserAgent:
        """Load the DiffusionDrive-Style model from checkpoint."""
        from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
        
        # Create default config for DiffusionDrive-Style
        config = TransfuserConfig()
        
        # Enable style support
        config.with_style = True
        config.styletrain_path = ""  # Not needed for inference
        config.styletest_path = ""   # Not needed for inference
        
        # Initialize agent with checkpoint
        agent = TransfuserAgent(
            config=config,
            lr=1e-4,  # Not used for inference
            checkpoint_path=str(self.checkpoint_path)
        )
        
        # Set to evaluation mode
        agent.eval()
        agent.to(self.device)
        
        return agent
        
    def _build_style_feature(self, style_name: str) -> torch.Tensor:
        """
        Build one-hot style feature vector.
        
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
        
    def predict_single_style(self, agent_input: AgentInput, style_name: str) -> torch.Tensor:
        """
        Predict trajectory for a single driving style.
        
        Args:
            agent_input: Input data for the scene
            style_name: Driving style ('aggressive', 'normal', 'conservative')
            
        Returns:
            Predicted trajectory tensor
        """
        self.agent.eval()
        
        # Build features using the agent's feature builders
        features = {}
        for builder in self.agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        
        # Add style feature
        style_feature = self._build_style_feature(style_name)
        features["style_feature"] = style_feature
        
        # Move features to device
        features = {k: v.to(self.device) for k, v in features.items()}
        
        # Run inference
        with torch.no_grad():
            predictions = self.agent.forward(features)
            trajectory = predictions["trajectory"]  # Shape: [batch_size, num_poses, 3]
            
        return trajectory
        
    def predict_all_styles(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        Predict trajectories for all three driving styles.
        
        Args:
            agent_input: Input data for the scene
            
        Returns:
            Dictionary mapping style names to predicted trajectories
        """
        results = {}
        
        for style_name in self.STYLE_MAP.keys():
            trajectory = self.predict_single_style(agent_input, style_name)
            results[style_name] = trajectory
            
        return results
        
    def get_style_info(self, style_name: str) -> Dict[str, Any]:
        """Get information about a specific style."""
        return self.STYLE_MAP.get(style_name, {})
        
    def get_available_styles(self) -> list:
        """Get list of available driving styles."""
        return list(self.STYLE_MAP.keys())