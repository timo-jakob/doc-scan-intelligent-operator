"""Tests for path utilities."""

import pytest
from pathlib import Path
from docscan.path_utils import validate_safe_path


def test_validate_safe_path_existing_file(tmp_path):
    """Test validating an existing file path."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    result = validate_safe_path(test_file, must_exist=True)
    assert result == test_file.resolve()


def test_validate_safe_path_nonexistent_file_must_exist(tmp_path):
    """Test validating a nonexistent file when it must exist."""
    test_file = tmp_path / "nonexistent.txt"

    with pytest.raises(FileNotFoundError):
        validate_safe_path(test_file, must_exist=True)


def test_validate_safe_path_nonexistent_file_not_required(tmp_path):
    """Test validating a nonexistent file when existence is not required."""
    test_file = tmp_path / "nonexistent.txt"

    result = validate_safe_path(test_file, must_exist=False)
    assert result == test_file.resolve()


def test_validate_safe_path_with_null_bytes():
    """Test that paths with null bytes are rejected."""
    with pytest.raises(ValueError, match="null bytes"):
        validate_safe_path(Path("test\0file.txt"))


def test_validate_safe_path_resolves_relative():
    """Test that relative paths are resolved to absolute paths."""
    result = validate_safe_path(Path("."), must_exist=True)
    assert result.is_absolute()


def test_validate_safe_path_resolves_parent_references(tmp_path):
    """Test that .. references are resolved."""
    test_dir = tmp_path / "subdir"
    test_dir.mkdir()
    test_file = test_dir / ".." / "test.txt"
    (tmp_path / "test.txt").write_text("content")

    result = validate_safe_path(test_file, must_exist=True)
    assert result == (tmp_path / "test.txt").resolve()
    assert ".." not in str(result)
