"""Tests for file renaming utilities."""

from pathlib import Path
import pytest

from docscan.file_renamer import rename_invoice


def test_rename_invoice_same_directory(tmp_path):
    """Test renaming file in same directory."""
    # Create test file
    original = tmp_path / "old_name.pdf"
    original.write_text("test content")

    # Rename file
    result = rename_invoice(original, "new_name.pdf")

    assert result is not None
    assert result.name == "new_name.pdf"
    assert result.exists()
    assert not original.exists()
    assert result.read_text() == "test content"


def test_rename_invoice_to_output_directory(tmp_path):
    """Test renaming file to different output directory."""
    # Create test file
    original = tmp_path / "invoice.pdf"
    original.write_text("test content")

    # Create output directory
    output_dir = tmp_path / "output"

    # Rename file
    result = rename_invoice(original, "renamed.pdf", output_dir=output_dir)

    assert result is not None
    assert result.parent == output_dir
    assert result.name == "renamed.pdf"
    assert result.exists()
    assert original.exists()  # Original should still exist (copy, not move)
    assert result.read_text() == "test content"


def test_rename_invoice_nonexistent_file(tmp_path):
    """Test renaming nonexistent file."""
    original = tmp_path / "nonexistent.pdf"

    result = rename_invoice(original, "new_name.pdf")

    assert result is None


def test_rename_invoice_dry_run(tmp_path):
    """Test dry run mode."""
    # Create test file
    original = tmp_path / "invoice.pdf"
    original.write_text("test content")

    # Dry run
    result = rename_invoice(original, "new_name.pdf", dry_run=True)

    assert result is None
    assert original.exists()  # Original should still exist
    assert not (tmp_path / "new_name.pdf").exists()  # New file should not be created


def test_rename_invoice_collision_handling(tmp_path):
    """Test handling of filename collisions."""
    # Create original file
    original = tmp_path / "invoice.pdf"
    original.write_text("original content")

    # Create existing file with target name
    existing = tmp_path / "target.pdf"
    existing.write_text("existing content")

    # Rename should create alternate filename
    result = rename_invoice(original, "target.pdf")

    assert result is not None
    assert result.name == "target_1.pdf"  # Should have counter suffix
    assert result.exists()
    assert existing.exists()  # Existing file should not be overwritten
    assert result.read_text() == "original content"
    assert existing.read_text() == "existing content"


def test_rename_invoice_multiple_collisions(tmp_path):
    """Test handling of multiple filename collisions."""
    # Create original file
    original = tmp_path / "invoice.pdf"
    original.write_text("original")

    # Create existing files
    (tmp_path / "target.pdf").write_text("existing1")
    (tmp_path / "target_1.pdf").write_text("existing2")
    (tmp_path / "target_2.pdf").write_text("existing3")

    # Rename should find next available number
    result = rename_invoice(original, "target.pdf")

    assert result is not None
    assert result.name == "target_3.pdf"
    assert result.exists()


def test_rename_invoice_creates_output_directory(tmp_path):
    """Test that output directory is created if it doesn't exist."""
    # Create test file
    original = tmp_path / "invoice.pdf"
    original.write_text("test content")

    # Use non-existent output directory
    output_dir = tmp_path / "deep" / "nested" / "path"

    # Rename file
    result = rename_invoice(original, "renamed.pdf", output_dir=output_dir)

    assert result is not None
    assert output_dir.exists()
    assert result.parent == output_dir
    assert result.exists()


def test_rename_invoice_preserves_extension(tmp_path):
    """Test that file extension is preserved."""
    # Create test file
    original = tmp_path / "invoice.pdf"
    original.write_text("test content")

    # Rename with new name that includes extension
    result = rename_invoice(original, "2024-01-15_Rechnung_Company.pdf")

    assert result is not None
    assert result.suffix == ".pdf"
    assert result.name == "2024-01-15_Rechnung_Company.pdf"


def test_rename_invoice_same_name_same_location(tmp_path):
    """Test renaming file to same name in same location."""
    # Create test file
    original = tmp_path / "invoice.pdf"
    original.write_text("test content")

    # Rename to same name
    result = rename_invoice(original, "invoice.pdf")

    # Should succeed without creating duplicate
    assert result is not None
    assert result == original
    assert result.exists()


def test_rename_invoice_permission_error(tmp_path, monkeypatch):
    """Test handling of permission errors."""
    # Create test file
    original = tmp_path / "invoice.pdf"
    original.write_text("test content")

    # Mock rename to raise PermissionError
    def mock_rename(self, target):
        raise PermissionError("No permission")

    monkeypatch.setattr(Path, "rename", mock_rename)

    # Rename should fail gracefully
    result = rename_invoice(original, "new_name.pdf")

    assert result is None
    assert original.exists()


def test_rename_invoice_with_spaces(tmp_path):
    """Test renaming with spaces in filename."""
    # Create test file
    original = tmp_path / "invoice.pdf"
    original.write_text("test content")

    # Rename with spaces
    result = rename_invoice(original, "My Invoice File.pdf")

    assert result is not None
    assert result.name == "My Invoice File.pdf"
    assert result.exists()


def test_rename_invoice_unicode_characters(tmp_path):
    """Test renaming with unicode characters."""
    # Create test file
    original = tmp_path / "invoice.pdf"
    original.write_text("test content")

    # Rename with unicode
    result = rename_invoice(original, "Rechnung_Müller_&_Söhne.pdf")

    assert result is not None
    assert result.name == "Rechnung_Müller_&_Söhne.pdf"
    assert result.exists()
