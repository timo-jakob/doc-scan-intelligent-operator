"""Configuration management for document scanner."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


DEFAULT_CONFIG = {
    # Vision-Language Model for invoice detection
    "vlm_model": "Qwen/Qwen2-VL-7B-Instruct",  # Default VLM for document analysis

    # Model cache directory
    "model_cache_dir": None,  # None = use default (~/.cache/docscan/models)

    # VLM generation parameters
    "vlm_config": {
        "max_tokens": 200,
        "temperature": 0.1,
    },

    # Supported document formats (currently only PDF)
    "supported_formats": [".pdf"],
}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from file or return defaults.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()

    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            if user_config:
                config.update(user_config)

    return config


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
