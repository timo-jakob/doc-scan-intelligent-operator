"""Model management for MLX-based inference."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Any
import json

from docscan.auth import login_to_huggingface
from docscan.mlx_utils import (
    check_mlx_compatibility,
    check_disk_space_for_model,
    format_bytes,
)

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages MLX model loading, caching, and lifecycle."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize model manager.

        Args:
            cache_dir: Directory for caching models. If None, uses default.
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "docscan" / "models"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track loaded models to avoid reloading
        self._loaded_models = {}

        logger.info(f"Model manager initialized with cache directory: {self.cache_dir}")

    def load_model(self, model_id: str, is_vlm: bool = False) -> Tuple[Any, Any]:
        """
        Load a model and tokenizer/processor for MLX inference.

        This method:
        1. Checks if model is already loaded (returns cached version)
        2. Validates MLX compatibility
        3. Checks disk space
        4. Authenticates to HuggingFace if needed
        5. Downloads and loads the model with MLX optimization

        Args:
            model_id: HuggingFace model identifier
            is_vlm: Whether this is a vision-language model

        Returns:
            Tuple of (model, processor) where processor is tokenizer for text models
            or image processor for VLMs

        Raises:
            ValueError: If model is not MLX compatible or insufficient disk space
            RuntimeError: If model loading fails
        """
        # Check if already loaded
        if model_id in self._loaded_models:
            logger.info(f"Using cached model: {model_id}")
            return self._loaded_models[model_id]

        # Check MLX compatibility
        logger.info(f"Checking MLX compatibility for {model_id}")
        is_compatible, reason = check_mlx_compatibility(model_id, is_vlm=is_vlm)
        if not is_compatible:
            raise ValueError(f"Model {model_id} is not compatible with MLX: {reason}")

        # Check disk space
        logger.info(f"Checking disk space for {model_id}")
        has_space, message = check_disk_space_for_model(model_id, self.cache_dir)
        if not has_space:
            raise ValueError(message)

        # Authenticate to HuggingFace
        login_to_huggingface()

        # Load model with MLX
        logger.info(f"Loading model {model_id} with MLX optimization...")
        try:
            if is_vlm:
                model, processor = self._load_mlx_vlm(model_id)
            else:
                model, processor = self._load_mlx_model(model_id)

            logger.info(f"Successfully loaded {model_id}")

            # Cache the loaded model
            self._loaded_models[model_id] = (model, processor)

            # Save model metadata
            self._save_model_metadata(model_id)

            return model, processor

        except (ImportError, RuntimeError, OSError, ValueError) as e:
            # Catches import errors, MLX errors, file errors, model loading errors
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Failed to load model {model_id}: {e}")

    def _load_mlx_model(self, model_id: str) -> Tuple[Any, Any]:
        """
        Load text model using MLX.

        Args:
            model_id: HuggingFace model identifier

        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            import mlx.core as mx
            from mlx_lm import load

            # Load model and tokenizer with MLX
            # The load function from mlx_lm handles downloading and caching
            model, tokenizer = load(
                model_id,
                tokenizer_config={"trust_remote_code": True}
            )

            logger.info(f"Text model {model_id} loaded on MLX device")
            return model, tokenizer

        except ImportError as e:
            raise RuntimeError(
                "MLX packages not installed. Please install: pip install mlx mlx-lm"
            ) from e
        except (OSError, ValueError, RuntimeError) as e:
            # Catches file errors, model format errors, and MLX runtime errors
            raise RuntimeError(f"Failed to load MLX model: {e}") from e

    def _load_mlx_vlm(self, model_id: str) -> Tuple[Any, Any]:
        """
        Load vision-language model using MLX.

        Args:
            model_id: HuggingFace model identifier

        Returns:
            Tuple of (model, processor)
        """
        try:
            import mlx.core as mx
            from mlx_vlm import load

            # Load VLM model and processor with MLX
            # The load function from mlx_vlm handles downloading and caching
            model, processor = load(
                model_id,
                processor_config={"trust_remote_code": True}
            )

            logger.info(f"VLM {model_id} loaded on MLX device")
            return model, processor

        except ImportError as e:
            raise RuntimeError(
                "MLX VLM packages not installed. Please install: pip install mlx mlx-vlm"
            ) from e
        except (OSError, ValueError, RuntimeError) as e:
            # Catches file errors, model format errors, and MLX runtime errors
            raise RuntimeError(f"Failed to load MLX VLM: {e}") from e

    def _save_model_metadata(self, model_id: str) -> None:
        """
        Save metadata about loaded model.

        Args:
            model_id: Model identifier
        """
        metadata_file = self.cache_dir / "model_metadata.json"

        try:
            # Load existing metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Add/update entry
            from datetime import datetime
            metadata[model_id] = {
                "loaded_at": datetime.now().isoformat(),
                "cache_dir": str(self.cache_dir),
            }

            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        except (OSError, PermissionError, json.JSONDecodeError) as e:
            # Catches file write errors and JSON errors
            logger.warning(f"Failed to save model metadata: {e}")

    def get_cached_models(self) -> list:
        """
        Get list of cached model identifiers.

        Returns:
            List of model IDs that have been cached
        """
        metadata_file = self.cache_dir / "model_metadata.json"

        if not metadata_file.exists():
            return []

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return list(metadata.keys())
        except (OSError, PermissionError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read model metadata: {e}")
            return []

    def clear_cache(self, model_id: Optional[str] = None) -> None:
        """
        Clear model cache.

        Args:
            model_id: Specific model to remove, or None to clear all
        """
        if model_id:
            # Remove specific model from loaded cache
            if model_id in self._loaded_models:
                del self._loaded_models[model_id]
                logger.info(f"Cleared cached model: {model_id}")
        else:
            # Clear all loaded models
            self._loaded_models.clear()
            logger.info("Cleared all cached models")

    def get_cache_info(self) -> dict:
        """
        Get information about the model cache.

        Returns:
            Dictionary with cache information
        """
        from docscan.mlx_utils import get_available_disk_space, format_bytes

        cache_size = 0
        model_count = 0

        # Calculate cache directory size
        if self.cache_dir.exists():
            for item in self.cache_dir.rglob("*"):
                if item.is_file():
                    try:
                        cache_size += item.stat().st_size
                        model_count += 1
                    except OSError:
                        # Skip files we can't access
                        pass

        available_space = get_available_disk_space(self.cache_dir)

        return {
            "cache_dir": str(self.cache_dir),
            "cache_size": format_bytes(cache_size),
            "cache_size_bytes": cache_size,
            "file_count": model_count,
            "available_space": format_bytes(available_space),
            "available_space_bytes": available_space,
            "loaded_models": list(self._loaded_models.keys()),
            "cached_models": self.get_cached_models(),
        }
