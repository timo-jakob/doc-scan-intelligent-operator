"""MLX compatibility checking and model utilities."""

import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_valid_pipeline_tag(pipeline_tag: str, is_vlm: bool) -> bool:
    """Check if pipeline tag is valid for the model type."""
    if is_vlm:
        valid_tags = ['image-text-to-text', 'visual-question-answering', 'text-generation']
        return not pipeline_tag or pipeline_tag in valid_tags
    else:
        return pipeline_tag in ['text-generation', 'text2text-generation']


def _get_mlx_compatible_architectures() -> list:
    """Return list of known MLX-compatible architectures."""
    return [
        'llama',
        'mistral',
        'phi',
        'qwen',
        'qwen2',
        'qwen2_vl',  # VLM
        'mllama',  # VLM - Llama 3.2 Vision
        'gpt2',
        'gpt_neox',
        'stablelm',
        'mixtral',
        'llava',  # VLM
        'llava_next',  # VLM
        'idefics',  # VLM
        'paligemma',  # VLM
    ]


def check_mlx_compatibility(model_id: str, is_vlm: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Check if a model is compatible with MLX.

    MLX-compatible models typically:
    - Are text generation or vision-language models
    - Have supported architectures (Llama, Mistral, Phi, Qwen-VL, LLaVA, etc.)
    - Don't require special hardware beyond Apple Silicon

    Args:
        model_id: HuggingFace model identifier
        is_vlm: Whether this is a vision-language model

    Returns:
        Tuple of (is_compatible, reason_if_not)
    """
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)

        # Check pipeline tag
        if hasattr(info, 'pipeline_tag') and info.pipeline_tag:
            if not _is_valid_pipeline_tag(info.pipeline_tag, is_vlm):
                return False, f"Model is {info.pipeline_tag}, not text-generation"

        # Check model architecture from config
        if hasattr(info, 'config') and info.config:
            model_type = info.config.get('model_type', '').lower()
            mlx_architectures = _get_mlx_compatible_architectures()
            
            if model_type and model_type not in mlx_architectures:
                return False, (
                    f"Architecture '{model_type}' may not be supported by MLX. "
                    f"Supported: {', '.join(mlx_architectures)}"
                )

        logger.info(f"Model {model_id} appears to be MLX-compatible")
        return True, None

    except ImportError:
        logger.error("huggingface_hub not installed")
        return False, "huggingface_hub package not installed"
    except (OSError, ValueError, RuntimeError) as e:
        # Catches file errors, network errors, API errors, and HF hub errors
        logger.warning(f"Could not verify MLX compatibility for {model_id}: {e}")
        # If we can't check, assume it might work
        return True, None


def estimate_model_size(model_id: str) -> Optional[int]:
    """
    Estimate the size of a model in bytes.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        Estimated size in bytes, or None if cannot determine
    """
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)

        # Try to get size from safetensors or model files
        total_size = 0
        if hasattr(info, 'siblings') and info.siblings:
            for file in info.siblings:
                if hasattr(file, 'size') and file.size:
                    total_size += file.size

        if total_size > 0:
            return total_size

        # Fallback: estimate based on parameter count
        # Rough estimate: ~4 bytes per parameter for FP32, ~2 for FP16
        if hasattr(info, 'config') and info.config:
            # Try to parse parameter count from model card or config
            if 'num_parameters' in info.config:
                params = info.config['num_parameters']
                # Assume FP16 (2 bytes per param) + some overhead
                return int(params * 2.5)

        # Try to parse from model name (e.g., "7B" in model name)
        import re
        # Use a safe, specific regex pattern to avoid ReDoS
        match = re.search(r'(\d{1,3})B', model_id, re.IGNORECASE)
        if match:
            billions = int(match.group(1))
            # Rough estimate: 7B model ~14GB, scale linearly
            return int(billions * 2 * 1024 * 1024 * 1024)

        logger.warning(f"Could not estimate size for {model_id}")
        return None

    except (OSError, ValueError, RuntimeError) as e:
        # Catches file errors, network errors, API errors, and parsing errors
        logger.warning(f"Failed to estimate model size: {e}")
        return None


def get_available_disk_space(path: Path) -> int:
    """
    Get available disk space at the given path in bytes.

    Args:
        path: Path to check (file or directory)

    Returns:
        Available space in bytes
    """
    import shutil

    # Ensure we're checking a directory
    if path.is_file():
        path = path.parent

    stat = shutil.disk_usage(path)
    return stat.free


def format_bytes(bytes_size: int) -> str:
    """
    Format bytes into human-readable string.

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def check_disk_space_for_model(model_id: str, cache_dir: Path) -> Tuple[bool, str]:
    """
    Check if there's enough disk space to download a model.

    Args:
        model_id: HuggingFace model identifier
        cache_dir: Directory where model will be cached

    Returns:
        Tuple of (has_space, message)
    """
    # Ensure cache directory exists or can be created
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"Cannot create cache directory {cache_dir}: {e}"

    available_space = get_available_disk_space(cache_dir)
    estimated_size = estimate_model_size(model_id)

    if estimated_size is None:
        # If we can't estimate, warn but allow
        logger.warning(f"Could not estimate size for {model_id}, proceeding anyway")
        return True, "Could not estimate model size, proceeding with download"

    # Require 20% buffer beyond estimated size
    required_space = int(estimated_size * 1.2)

    if available_space < required_space:
        return False, (
            f"Insufficient disk space to download model '{model_id}'.\n"
            f"  Location: {cache_dir}\n"
            f"  Available: {format_bytes(available_space)}\n"
            f"  Required: {format_bytes(required_space)} (estimated)\n"
            f"  Model size: ~{format_bytes(estimated_size)}\n"
            f"Please free up at least {format_bytes(required_space - available_space)} "
            f"or configure a different cache directory."
        )

    logger.info(
        f"Disk space check passed: {format_bytes(available_space)} available, "
        f"~{format_bytes(estimated_size)} needed for {model_id}"
    )
    return True, "Sufficient disk space available"
