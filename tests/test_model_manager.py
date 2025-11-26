"""Tests for model manager."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from docscan.model_manager import ModelManager


def test_model_manager_initialization(tmp_path):
    """Test model manager initialization."""
    cache_dir = tmp_path / "models"
    manager = ModelManager(cache_dir)

    assert manager.cache_dir == cache_dir
    assert cache_dir.exists()
    assert manager._loaded_models == {}


def test_model_manager_default_cache_dir():
    """Test model manager uses default cache directory."""
    manager = ModelManager()
    expected_dir = Path.home() / ".cache" / "docscan" / "models"
    assert manager.cache_dir == expected_dir


def test_get_cached_models_empty(tmp_path):
    """Test getting cached models when none exist."""
    manager = ModelManager(tmp_path)
    cached = manager.get_cached_models()
    assert cached == []


def test_save_and_get_model_metadata(tmp_path):
    """Test saving and retrieving model metadata."""
    manager = ModelManager(tmp_path)

    # Save metadata
    manager._save_model_metadata("test/model")

    # Retrieve metadata
    cached = manager.get_cached_models()
    assert "test/model" in cached


def test_clear_cache_specific_model(tmp_path):
    """Test clearing specific model from cache."""
    manager = ModelManager(tmp_path)
    manager._loaded_models = {
        "model1": ("mock_model1", "mock_tokenizer1"),
        "model2": ("mock_model2", "mock_tokenizer2"),
    }

    manager.clear_cache("model1")
    assert "model1" not in manager._loaded_models
    assert "model2" in manager._loaded_models


def test_clear_cache_all(tmp_path):
    """Test clearing all models from cache."""
    manager = ModelManager(tmp_path)
    manager._loaded_models = {
        "model1": ("mock_model1", "mock_tokenizer1"),
        "model2": ("mock_model2", "mock_tokenizer2"),
    }

    manager.clear_cache()
    assert manager._loaded_models == {}


def test_get_cache_info(tmp_path):
    """Test getting cache information."""
    manager = ModelManager(tmp_path)

    # Create some test files
    (tmp_path / "test_file.bin").write_bytes(b"x" * 1000)

    manager._loaded_models = {"test/model": ("mock", "mock")}

    info = manager.get_cache_info()

    assert info["cache_dir"] == str(tmp_path)
    assert "cache_size" in info
    assert "available_space" in info
    assert info["loaded_models"] == ["test/model"]


@patch("docscan.model_manager.check_mlx_compatibility")
@patch("docscan.model_manager.check_disk_space_for_model")
@patch("docscan.model_manager.login_to_huggingface")
def test_load_model_checks_compatibility(
    mock_login, mock_disk_check, mock_compat_check, tmp_path
):
    """Test that load_model checks MLX compatibility."""
    manager = ModelManager(tmp_path)

    # Model not compatible
    mock_compat_check.return_value = (False, "Not compatible")

    with pytest.raises(ValueError, match="Not compatible"):
        manager.load_model("test/model")


@patch("docscan.model_manager.check_mlx_compatibility")
@patch("docscan.model_manager.check_disk_space_for_model")
@patch("docscan.model_manager.login_to_huggingface")
def test_load_model_checks_disk_space(
    mock_login, mock_disk_check, mock_compat_check, tmp_path
):
    """Test that load_model checks disk space."""
    manager = ModelManager(tmp_path)

    mock_compat_check.return_value = (True, None)
    mock_disk_check.return_value = (False, "Insufficient space")

    with pytest.raises(ValueError, match="Insufficient space"):
        manager.load_model("test/model")


@patch("docscan.model_manager.check_mlx_compatibility")
@patch("docscan.model_manager.check_disk_space_for_model")
@patch("docscan.model_manager.login_to_huggingface")
@patch("docscan.model_manager.ModelManager._load_mlx_model")
def test_load_model_caches_loaded_model(
    mock_load_mlx, mock_login, mock_disk_check, mock_compat_check, tmp_path
):
    """Test that loaded models are cached."""
    manager = ModelManager(tmp_path)

    mock_compat_check.return_value = (True, None)
    mock_disk_check.return_value = (True, "OK")
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load_mlx.return_value = (mock_model, mock_tokenizer)

    # First load
    model1, tokenizer1 = manager.load_model("test/model")
    assert mock_load_mlx.call_count == 1

    # Second load should use cache
    model2, tokenizer2 = manager.load_model("test/model")
    assert mock_load_mlx.call_count == 1  # Not called again
    assert model1 is model2
    assert tokenizer1 is tokenizer2
