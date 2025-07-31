"""
Configuration module for StyleTrajectoryApp
"""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_default_config() -> Dict[str, Any]:
    """Load the default configuration"""
    config_path = Path(__file__).parent / "default_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

__all__ = ["load_default_config", "load_config"]