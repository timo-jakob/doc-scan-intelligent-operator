"""Tests for HuggingFace authentication."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from docscan.auth import get_huggingface_token, login_to_huggingface


def test_get_token_from_env_hf_token(monkeypatch):
    """Test getting token from HF_TOKEN environment variable."""
    monkeypatch.setenv("HF_TOKEN", "test_token_123")
    token = get_huggingface_token()
    assert token == "test_token_123"


def test_get_token_from_env_hugging_face_token(monkeypatch):
    """Test getting token from HUGGING_FACE_TOKEN environment variable."""
    monkeypatch.setenv("HUGGING_FACE_TOKEN", "test_token_456")
    token = get_huggingface_token()
    assert token == "test_token_456"


def test_get_token_priority(monkeypatch):
    """Test that HF_TOKEN has priority over HUGGING_FACE_TOKEN."""
    monkeypatch.setenv("HF_TOKEN", "priority_token")
    monkeypatch.setenv("HUGGING_FACE_TOKEN", "secondary_token")
    token = get_huggingface_token()
    assert token == "priority_token"


def test_get_token_from_file(tmp_path, monkeypatch):
    """Test getting token from file."""
    # Clear environment variables
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)

    # Create token file in home directory
    token_file = tmp_path / ".huggingface" / "token"
    token_file.parent.mkdir(parents=True)
    token_file.write_text("file_token_789\n")

    # Mock Path.home() to return our tmp_path
    with patch("pathlib.Path.home", return_value=tmp_path):
        token = get_huggingface_token()
        assert token == "file_token_789"


def test_get_token_none_when_not_found(monkeypatch):
    """Test that None is returned when no token is found."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)

    with patch("pathlib.Path.home", return_value=Path("/nonexistent")):
        token = get_huggingface_token()
        assert token is None


@patch("huggingface_hub.login")
def test_login_success(mock_login, monkeypatch):
    """Test successful login to HuggingFace."""
    monkeypatch.setenv("HF_TOKEN", "test_token")
    mock_login.return_value = None

    result = login_to_huggingface()
    assert result is True
    mock_login.assert_called_once_with(token="test_token", add_to_git_credential=False)


@patch("huggingface_hub.login")
def test_login_with_explicit_token(mock_login):
    """Test login with explicitly provided token."""
    mock_login.return_value = None

    result = login_to_huggingface(token="explicit_token")
    assert result is True
    mock_login.assert_called_once_with(token="explicit_token", add_to_git_credential=False)


def test_login_without_token(monkeypatch):
    """Test login fails gracefully when no token available."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)

    with patch("pathlib.Path.home", return_value=Path("/nonexistent")):
        result = login_to_huggingface()
        assert result is False


def test_get_token_from_file_permission_error(tmp_path, monkeypatch):
    """Test handling permission error when reading token file."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)

    # Create token file
    token_file = tmp_path / ".huggingface" / "token"
    token_file.parent.mkdir(parents=True)
    token_file.write_text("test_token")

    # Mock the read_text to raise PermissionError
    with patch("pathlib.Path.home", return_value=tmp_path):
        with patch("pathlib.Path.read_text", side_effect=PermissionError("Access denied")):
            token = get_huggingface_token()
            assert token is None


def test_login_import_error():
    """Test login fails gracefully when huggingface_hub not installed."""
    with patch.dict('sys.modules', {'huggingface_hub': None}):
        with patch("docscan.auth.get_huggingface_token", return_value="test_token"):
            result = login_to_huggingface()
            assert result is False


@patch("huggingface_hub.login")
def test_login_general_exception(mock_login):
    """Test login handles general exceptions during login."""
    mock_login.side_effect = RuntimeError("Login failed")

    result = login_to_huggingface(token="test_token")
    assert result is False
