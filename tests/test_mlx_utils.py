"""Tests for MLX utilities."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from docscan.mlx_utils import (
    check_mlx_compatibility,
    estimate_model_size,
    get_available_disk_space,
    format_bytes,
    check_disk_space_for_model,
)


def test_format_bytes():
    """Test byte formatting."""
    assert format_bytes(500) == "500.00 B"
    assert format_bytes(1024) == "1.00 KB"
    assert format_bytes(1024 * 1024) == "1.00 MB"
    assert format_bytes(1024 * 1024 * 1024) == "1.00 GB"
    assert format_bytes(5 * 1024 * 1024 * 1024) == "5.00 GB"


def test_get_available_disk_space(tmp_path):
    """Test getting available disk space."""
    space = get_available_disk_space(tmp_path)
    assert isinstance(space, int)
    assert space > 0


def test_get_available_disk_space_for_file(tmp_path):
    """Test getting disk space for a file path."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    space = get_available_disk_space(test_file)
    assert isinstance(space, int)
    assert space > 0


@patch("docscan.mlx_utils.model_info")
def test_check_mlx_compatibility_text_generation(mock_model_info):
    """Test MLX compatibility check for text generation model."""
    mock_info = MagicMock()
    mock_info.pipeline_tag = "text-generation"
    mock_info.config = {"model_type": "llama"}
    mock_model_info.return_value = mock_info

    is_compatible, reason = check_mlx_compatibility("test/model")
    assert is_compatible is True
    assert reason is None


@patch("docscan.mlx_utils.model_info")
def test_check_mlx_compatibility_wrong_pipeline(mock_model_info):
    """Test MLX compatibility check for non-text-generation model."""
    mock_info = MagicMock()
    mock_info.pipeline_tag = "image-classification"
    mock_model_info.return_value = mock_info

    is_compatible, reason = check_mlx_compatibility("test/model")
    assert is_compatible is False
    assert "image-classification" in reason


@patch("docscan.mlx_utils.model_info")
def test_check_mlx_compatibility_unsupported_architecture(mock_model_info):
    """Test MLX compatibility for unsupported architecture."""
    mock_info = MagicMock()
    mock_info.pipeline_tag = "text-generation"
    mock_info.config = {"model_type": "unsupported_arch"}
    mock_model_info.return_value = mock_info

    is_compatible, reason = check_mlx_compatibility("test/model")
    assert is_compatible is False
    assert "unsupported_arch" in reason


@patch("docscan.mlx_utils.model_info")
def test_estimate_model_size_from_files(mock_model_info):
    """Test model size estimation from file sizes."""
    mock_file1 = MagicMock()
    mock_file1.size = 1024 * 1024 * 100  # 100 MB
    mock_file2 = MagicMock()
    mock_file2.size = 1024 * 1024 * 200  # 200 MB

    mock_info = MagicMock()
    mock_info.siblings = [mock_file1, mock_file2]
    mock_model_info.return_value = mock_info

    size = estimate_model_size("test/model")
    assert size == 1024 * 1024 * 300  # 300 MB total


@patch("docscan.mlx_utils.model_info")
def test_estimate_model_size_from_name(mock_model_info):
    """Test model size estimation from model name."""
    mock_info = MagicMock()
    mock_info.siblings = []
    mock_info.config = {}
    mock_model_info.return_value = mock_info

    # Model name contains "7B"
    size = estimate_model_size("test/model-7B")
    assert size is not None
    assert size > 0
    # Should be approximately 14 GB (7B * 2 bytes)
    expected = 7 * 2 * 1024 * 1024 * 1024
    assert abs(size - expected) < expected * 0.1  # Within 10%


def test_check_disk_space_sufficient(tmp_path):
    """Test disk space check when sufficient space available."""
    # Create cache directory
    cache_dir = tmp_path / "cache"

    with patch("docscan.mlx_utils.estimate_model_size", return_value=1024 * 1024):  # 1 MB
        has_space, message = check_disk_space_for_model("test/model", cache_dir)
        assert has_space is True
        assert cache_dir.exists()


def test_check_disk_space_insufficient(tmp_path):
    """Test disk space check when insufficient space available."""
    cache_dir = tmp_path / "cache"

    # Mock very large model size
    huge_size = 1024 * 1024 * 1024 * 1024 * 1024  # 1 PB

    with patch("docscan.mlx_utils.estimate_model_size", return_value=huge_size):
        has_space, message = check_disk_space_for_model("test/model", cache_dir)
        assert has_space is False
        assert "test/model" in message
        assert "Insufficient disk space" in message
        assert str(cache_dir) in message


def test_check_disk_space_unknown_size(tmp_path):
    """Test disk space check when model size cannot be estimated."""
    cache_dir = tmp_path / "cache"

    with patch("docscan.mlx_utils.estimate_model_size", return_value=None):
        has_space, message = check_disk_space_for_model("test/model", cache_dir)
        # Should allow when size cannot be estimated
        assert has_space is True
        assert "Could not estimate" in message
