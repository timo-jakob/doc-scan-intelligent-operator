"""Tests for configuration management."""

import pytest
from pathlib import Path
from docscan.config import load_config, save_config, DEFAULT_CONFIG


def test_load_default_config():
    """Test loading default configuration."""
    config = load_config()
    assert config == DEFAULT_CONFIG
    assert "vlm_model" in config
    assert "supported_formats" in config
    assert "vlm_config" in config


def test_load_config_from_file(tmp_path):
    """Test loading configuration from YAML file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
model: custom-model
output_dir: /custom/path
categories:
  - invoice
  - receipt
""")

    config = load_config(config_file)
    assert config["model"] == "custom-model"
    assert config["output_dir"] == "/custom/path"
    assert len(config["categories"]) == 2


def test_save_config(tmp_path):
    """Test saving configuration to file."""
    config = {
        "model": "test-model",
        "output_dir": "/test/path"
    }
    output_file = tmp_path / "output_config.yaml"

    save_config(config, output_file)
    assert output_file.exists()

    # Load and verify
    loaded_config = load_config(output_file)
    assert loaded_config["model"] == "test-model"
