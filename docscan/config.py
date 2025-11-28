"""Configuration management for document scanner."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from docscan.path_utils import validate_safe_path


# Default model identifiers
DEFAULT_VLM_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_TEXT_LLM_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


DEFAULT_CONFIG = {
    # Vision-Language Model for invoice detection
    "vlm_model": DEFAULT_VLM_MODEL,

    # Text-only LLM for OCR-based analysis (lighter weight alternative)
    "text_llm_model": DEFAULT_TEXT_LLM_MODEL,

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

    if config_path:
        # Validate and resolve path to prevent path traversal
        config_path = validate_safe_path(config_path, must_exist=True)
        
        # Check if it's a file
        if not config_path.is_file():
            raise ValueError(f"Configuration path is not a file: {config_path}")
        
        # Ensure file has safe extension
        if config_path.suffix.lower() not in ['.yaml', '.yml']:
            raise ValueError(f"Configuration file must be YAML (.yaml or .yml): {config_path}")
        
        # Use file descriptor for safer file access
        with open(config_path, 'r', encoding='utf-8') as f:
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
    # Validate and resolve path to prevent path traversal
    output_path = validate_safe_path(output_path, must_exist=False)
    
    # Ensure file has safe extension
    if output_path.suffix.lower() not in ['.yaml', '.yml']:
        raise ValueError(f"Configuration file must be YAML (.yaml or .yml): {output_path}")
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
